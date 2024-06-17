from collections import deque
import copy
import os
import random
import sys
import math
import pickle
import time
from typing import List

import numpy
import torch
from examples.prefetch.client_manager import PrefetchClientManager
from examples.prefetch.constants import *
from examples.prefetch.utils import is_batch_norm_layer
from fedscale.cloud.aggregation.optimizers import TorchServerOptimizer
import fedscale.cloud.channels.job_api_pb2 as job_api_pb2
from fedscale.cloud import commons
from fedscale.cloud.aggregation.aggregator import Aggregator
from fedscale.cloud.internal.torch_model_adapter import TorchModelAdapter
from fedscale.cloud.logger.aggregator_logging import *
from fedscale.utils.compressor import Compressor
from fedscale.utils.compressor.identity import IdentityCompressor
from fedscale.utils.compressor.lfl import LFLCompressor
from fedscale.utils.compressor.qsgd import QSGDCompressor
from fedscale.utils.compressor.qsgd_bucket import QSGDBucketCompressor
from fedscale.utils.compressor.topk import TopKCompressor
from fedscale.utils.eval.round_evaluator import RoundEvaluator
from fedscale.utils.eval.sparsification import Sparsification


class PrefetchAggregator(Aggregator):
    """Feed aggregator using tensorflow models"""

    def __init__(self, args):
        super().__init__(args)
        self.client_manager = self.init_client_manager(args)

        self.feasible_client_count = 0 # Initialized after executors have finished client registration
        self.num_participants = args.num_participants

        self.total_mask_ratio = args.total_mask_ratio  # = shared_mask + local_mask
        self.shared_mask_ratio = args.shared_mask_ratio
        self.regenerate_epoch = args.regenerate_epoch
        self.max_prefetch_round = args.max_prefetch_round

        self.sampling_strategy = args.sampling_strategy
        self.sticky_group_size = args.sticky_group_size
        self.sticky_group_change_num = args.sticky_group_change_num
        self.sampled_sticky_client_set = []
        self.sampled_sticky_clients = []
        self.sampled_changed_clients = []

        self.fl_method = args.fl_method

        self.compressed_gradient = []
        self.mask_record_list = []
        self.shared_mask = []
        self.apf_ratio = 1.0

        self.last_update_index = []
        self.round_lost_clients = []
        self.clients_to_run = []
        self.slowest_client_id = -1
        self.round_evaluator = RoundEvaluator()

        # TODO Extract scheduler logic
        self.enable_prefetch = args.enable_prefetch
        self.max_prefetch_round = args.max_prefetch_round
        self.prefetch_estimation_start = args.prefetch_estimation_start
        self.sampled_clients = []
        self.sampled_sticky_clients = []
        self.sampled_changed_clients = []

        # self.compressed_gradient_ctx = []
        # self.cum_gradients = []
        self.last_round_compressed_gradient = []
        self.first_compressed_gradient = []
        self.first_state_dict = []

        self.last_round_decompressed_gradient = []

        self.download_compressor_type = args.download_compressor_type
        self.upload_compressor_type = args.upload_compressor_type
        self.prefetch_compressor_type = args.prefetch_compressor_type
        self.quantization_target = args.quantization_target

        self.prev_model_weights = None
        self.client_estimate_weights = None # i.e. \hat{\theta}(t) in LFL

        # TODO currently unused
        self.download_sparsification_type = args.download_sparsification_type
        self.upload_sparsification_type = args.upload_sparsification_type

    def init_model(self):
        """Initialize the model"""
        if self.args.engine != commons.PYTORCH:
            raise ValueError(f"{self.args.engine} is not a supported engine for prefetch")
        
        # TODO Decide whether to change to PrefetchModelAdaptor
        self.model_wrapper = TorchModelAdapter(
            init_model(),
            optimizer=TorchServerOptimizer(
                    self.args.gradient_policy, self.args, self.device
            ),
        )

        self.model_weights = self.model_wrapper.get_weights()

    def init_client_manager(self, args):
        """Initialize Prefetch FL client sampler

        Args:
            args (dictionary): Variable arguments for fedscale runtime config. defaults to the setup in arg_parser.py

        Returns:
            PrfetchClientManager: The client manager class
        """

        # sample_mode: random or oort
        client_manager = PrefetchClientManager(args.sample_mode, args=args)

        return client_manager

    def init_mask(self):
        self.shared_mask = []
        for idx, param in enumerate(self.model_wrapper.get_model().state_dict().values()):
            self.shared_mask.append(
                torch.zeros_like(param, dtype=torch.bool).to(dtype=torch.bool)
            )

    def client_register_handler(self, executorId, info):
        """Triggered once after all executors have registered

        Args:
            executorId (int): Executor Id
            info (dictionary): Executor information

        """
        logging.info(f"Loading {len(info['size'])} client traces ...")
        for _size in info["size"]:
            # since the worker rankId starts from 1, we also configure the initial dataId as 1
            mapped_id = (
                (self.num_of_clients + 1) % len(self.client_profiles)
                if len(self.client_profiles) > 0
                else 1
            )
            systemProfile = self.client_profiles.get(
                mapped_id,
                {
                    "computation": 1.0,
                    "communication": 1.0,
                    "dl_kbps": 1.0,
                    "ul_kbps": 1.0,
                },
            )

            client_id = (
                (self.num_of_clients + 1)
                if self.experiment_mode == commons.SIMULATION_MODE
                else executorId
            )
            self.client_manager.register_client(
                executorId, client_id, size=_size, speed=systemProfile
            )
            self.client_manager.registerDuration(
                client_id,
                batch_size=self.args.batch_size,
                local_steps=self.args.local_steps,
                upload_size=self.model_update_size,
                download_size=self.model_update_size,
            )
            self.num_of_clients += 1

        logging.info(
            "Info of all feasible clients {}".format(self.client_manager.getDataInfo())
        )

        # Post client registration aggregator initialization
        self.feasible_client_count = len(self.client_manager.feasibleClients)
        self.last_update_index = [
            -1 for _ in range(self.feasible_client_count * 2)
        ]

    def run(self):
        """Start running the aggregator server by setting up execution
        and communication environment, and monitoring the grpc message.
        """
        self.setup_env()
        self.client_profiles = self.load_client_profile(
            file_path=self.args.device_conf_file
        )

        self.init_control_communication()
        self.init_data_communication()

        self.init_model()
        self.init_mask()

        self.model_update_size = (
            sys.getsizeof(pickle.dumps(self.model_wrapper)) / 1024.0 * 8.0
        )  # kbits
        self.model_bitmap_size = self.model_update_size / 32


        # Quantization
        self.prev_model_weights = self.model_wrapper.get_weights_torch()
        self.client_estimate_weights = self.model_wrapper.get_weights_torch()
        self.quantized_update = None
        
        self.event_monitor()

    def get_shared_mask(self):
        """Get shared mask that would be used by all FL clients (in default FL)

        Returns:
            List of PyTorch tensor: Based on the executor's machine learning framework, initialize and return the mask for training.

        """
        return [p.to(device="cpu") for p in self.shared_mask]

    def tictak_client_tasks_old(self, sampled_clients, num_clients_to_collect):
        """Record sampled client execution information in last round. In the SIMULATION_MODE,
        further filter the sampled_client and pick the top num_clients_to_collect clients.

        Args:
            sampled_clients (list of int): Sampled clients from client manager
            num_clients_to_collect (int): The number of clients actually needed for next round.

        Returns:
            tuple: Return the sampled clients and client execution information in the last round.

        """

        if len(sampled_clients) == 0:
            return [], [], [], {}, 0, []

        if self.experiment_mode == commons.SIMULATION_MODE:
            # NOTE: We try to remove dummy events as much as possible in simulations,
            # by removing the stragglers/offline clients in overcommitment"""
            sampledClientsReal = []
            sampledClientsLost = []
            completionTimes = []
            virtual_client_clock = {}

            # prefetch stats
            prefetched_clients = set()

            # 1. remove dummy clients that are not available to the end of training
            for client_to_run in sampled_clients:
                client_cfg = self.client_conf.get(client_to_run, self.args)
                exe_cost = {
                    "computation": 0,
                    "downstream": 0,
                    "upstream": 0,
                    "round_duration": 0,
                }
                if self.fl_method == FEDAVG:
                    exe_cost = self.client_manager.get_completion_time(
                        client_to_run,
                        batch_size=client_cfg.batch_size,
                        local_steps=client_cfg.local_steps,
                        upload_size=self.model_update_size,
                        download_size=self.model_update_size,
                        in_bits=False
                    )
                    self.round_evaluator.record_client(
                        client_to_run, self.model_update_size, self.model_update_size, exe_cost
                    )
                elif self.fl_method == STC:
                    l = self.last_update_index[client_to_run]
                    r = self.round - 1
                    downstream_update_ratio = (
                        Sparsification.check_model_update_overhead(
                            l,
                            r,
                            self.model_wrapper.get_model(),
                            self.mask_record_list,
                            self.device,
                            use_accurate_cache=True,
                        )
                    )
                    dl_size = min(
                        self.model_update_size * downstream_update_ratio
                        + self.model_bitmap_size,
                        self.model_update_size,
                    )
                    ul_size = (
                        self.total_mask_ratio * self.model_update_size
                        + self.model_bitmap_size
                    )

                    exe_cost = self.client_manager.get_completion_time(
                        client_to_run,
                        batch_size=client_cfg.batch_size,
                        local_steps=client_cfg.local_steps,
                        upload_size=ul_size,
                        download_size=dl_size,
                        in_bits=False
                    )
                    self.round_evaluator.record_client(
                        client_to_run, dl_size, ul_size, exe_cost
                    )
                elif self.fl_method == GLUEFL:
                    l = self.last_update_index[client_to_run]
                    r = self.round - 1
                    downstream_update_ratio = (
                        Sparsification.check_model_update_overhead(
                            l,
                            r,
                            self.model_wrapper.get_model(),
                            self.mask_record_list,
                            self.device,
                            use_accurate_cache=True,
                        )
                    )
                    dl_size = min(
                        self.model_update_size * downstream_update_ratio
                        + self.model_bitmap_size,
                        self.model_update_size,
                    )
                    ul_size = self.total_mask_ratio * self.model_update_size + min(
                        (self.total_mask_ratio - self.shared_mask_ratio)
                        * self.model_update_size,
                        self.model_bitmap_size,
                    )

                    exe_cost = self.client_manager.get_completion_time(
                        client_to_run,
                        batch_size=client_cfg.batch_size,
                        local_steps=client_cfg.local_steps,
                        upload_size=ul_size,
                        download_size=dl_size,
                        in_bits=False
                    )
                    self.round_evaluator.record_client(
                        client_to_run, dl_size, ul_size, exe_cost
                    )
                elif self.fl_method in ["GlueFLPrefetchA", "GlueFLPrefetchB", "GlueFLPrefetchC", "STCPrefetch"]:
                    # This is an estimate by the server
                    can_fully_prefetch = False
                    prefetch_completed_round = 0
                    # These are the actual result of the prefetch
                    # 0 if participated recently, 1 if can fully prefetch, else (0, 1) if can partially prefetch.
                    prefetch_size = 0
                    prefetched_ratio = 0

                    logging.info(
                        f"Estimate prefetch client_id {client_to_run}, l {self.last_update_index[client_to_run]}, r {self.round - 1} and round {self.round}"
                    )
                    # logging.info(f"{client_to_run} is STICKY {client_to_run in self.previous_sampled_participants}")

                    prefetch_start_i = (
                        1
                        if self.args.per_client_prefetch
                        else max(
                            min(self.max_prefetch_round + 1, self.round - 1) - 1, 1
                        )
                    )
                    # Calculate backwards to see if client can finish prefetching in max_prefetch_round
                    for i in range(
                        prefetch_start_i,
                        min(self.max_prefetch_round + 1, self.round - 1),
                    ):
                        l, r = self.last_update_index[client_to_run], self.round - 1 - i
                        # logging.info(f"Estimate prefetch client_id {client_to_run}, l{l} and r{r}, prefetch by {i}")
                        if (
                            l >= r
                        ):  # This case usually happens when the client participated in training recently
                            logging.info(
                                f"Unable to prefetch because client {client_to_run} participated recently"
                            )
                            break

                        round_durations = (
                            self.round_evaluator.round_durations_aggregated[
                                max(
                                    0, self.round - 1 - i - self.max_prefetch_round
                                ) : self.round
                                - 1
                                - i
                            ]
                        )
                        min_round_duration = min(round_durations)

                        prefetch_update_ratio = (
                            Sparsification.check_model_update_overhead(
                                l,
                                r,
                                self.model_wrapper.get_model(),
                                self.mask_record_list,
                                self.device,
                                use_accurate_cache=True,
                            )
                        )

                        # An optimization, the shared mask will always be changed so there is no point in trying to transfer the model corresponding to the shared mask
                        prefetch_size = min(
                            self.model_update_size * (1 - self.shared_mask_ratio),
                            self.model_update_size
                            * prefetch_update_ratio
                            * (1 - self.shared_mask_ratio)
                            + self.model_bitmap_size,
                        )
                        # logging.info(f"Prefetch update ratio {prefetch_update_ratio} prefetch size {prefetch_size} model update {self.model_update_size}\nTerm 1 {self.model_update_size * (1 - self.shared_mask_ratio)} Term 2 {self.model_update_size * prefetch_update_ratio * (1 - self.shared_mask_ratio) + self.model_bitmap_size}")

                        temp_pre_round = (
                            self.client_manager.get_download_time(
                                client_to_run, prefetch_size
                            )
                            / min_round_duration
                        )
                        logging.info(
                            f"Prefetch l {l} r {r} used min round duration {min_round_duration}, required prefetch round {temp_pre_round},  all usable round durations {round_durations}"
                        )

                        prefetch_completed_round = self.round - 1 - i
                        prefetched_ratio = min(
                            sum(round_durations[-i:])
                            / self.client_manager.get_download_time(
                                client_to_run, prefetch_size, in_bits=False
                            ),
                            1,
                        )

                        if temp_pre_round <= i:
                            can_fully_prefetch = True
                            break

                    if self.fl_method == "STCPrefetch":
                        ul_size = (
                            self.total_mask_ratio * self.model_update_size
                            + self.model_bitmap_size
                        )
                    else:
                        ul_size = self.total_mask_ratio * self.model_update_size + min(
                            (self.total_mask_ratio - self.shared_mask_ratio)
                            * self.model_update_size,
                            self.model_bitmap_size,
                        )

                    if can_fully_prefetch:
                        l, r = prefetch_completed_round, self.round - 1
                        downstream_update_ratio = (
                            Sparsification.check_model_update_overhead(
                                l,
                                r,
                                self.model_wrapper.get_model(),
                                self.mask_record_list,
                                self.device,
                                use_accurate_cache=True,
                            )
                        )
                        dl_size = min(
                            self.model_update_size * downstream_update_ratio
                            + self.model_bitmap_size,
                            self.model_update_size,
                        )
                        exe_cost = self.client_manager.get_completion_time(
                            client_to_run,
                            batch_size=client_cfg.batch_size,
                            local_steps=client_cfg.local_steps,
                            upload_size=ul_size,
                            download_size=dl_size,
                            in_bits=False
                        )
                        self.round_evaluator.record_client(
                            client_to_run,
                            dl_size,
                            ul_size,
                            exe_cost,
                            prefetch_dl_size=prefetch_size,
                        )
                        prefetched_clients.add(client_to_run)
                        logging.info(
                            f"After prefetch, l {l} and r {r}      dl_size {dl_size}     prefetch_dize {prefetch_size}"
                        )
                    elif prefetch_completed_round > 0:
                        # Partially prefetched
                        # For case where the prefetch budget is not sufficient, but at least something has been fetched
                        # In this case, on start of the client's scheduled round:
                        # Finish prefetched portion gets an update equivalent to missing one round
                        # Unfinished portion gets an update equivalent to missing all the rounds since the client's last update
                        l_prefeteched, l_unprefeteched, r = (
                            prefetch_completed_round,
                            self.last_update_index[client_to_run],
                            self.round - 1,
                        )
                        prefetched_downstream_update_ratio = (
                            Sparsification.check_model_update_overhead(
                                l_prefeteched,
                                r,
                                self.model_wrapper.get_model(),
                                self.mask_record_list,
                                self.device,
                                use_accurate_cache=True,
                            )
                        )
                        unprefetched_downstream_update_ratio = (
                            Sparsification.check_model_update_overhead(
                                l_unprefeteched,
                                r,
                                self.model_wrapper.get_model(),
                                self.mask_record_list,
                                self.device,
                                use_accurate_cache=True,
                            )
                        )
                        dl_size = min(
                            self.model_update_size
                            * (
                                prefetched_downstream_update_ratio * prefetched_ratio
                                + unprefetched_downstream_update_ratio
                                * (1 - prefetched_ratio)
                            )
                            + self.model_bitmap_size,
                            self.model_update_size,
                        )

                        exe_cost = self.client_manager.get_completion_time(
                            client_to_run,
                            batch_size=client_cfg.batch_size,
                            local_steps=client_cfg.local_steps,
                            upload_size=ul_size,
                            download_size=dl_size,
                            in_bits=False
                        )
                        self.round_evaluator.record_client(
                            client_to_run,
                            dl_size,
                            ul_size,
                            exe_cost,
                            prefetch_dl_size=prefetch_size * prefetched_ratio,
                        )
                        logging.info(
                            f"Unable to fully prefetch, l_pre {l_prefeteched}, l_nopre {l_unprefeteched} and r {r}   dl_size {dl_size}  prefetch_ratio {prefetched_ratio}  prefetch_size {prefetch_size} reason {'sticky or initial' if 0 <= r - l_unprefeteched <= 1 else 'slow'}"
                        )

                    else:
                        # No prefetch case
                        l, r = self.last_update_index[client_to_run], self.round - 1
                        downstream_update_ratio = (
                            Sparsification.check_model_update_overhead(
                                l,
                                r,
                                self.model_wrapper.get_model(),
                                self.mask_record_list,
                                self.device,
                                use_accurate_cache=True,
                            )
                        )
                        dl_size = min(
                            self.model_update_size * downstream_update_ratio
                            + self.model_bitmap_size,
                            self.model_update_size,
                        )
                        exe_cost = self.client_manager.get_completion_time(
                            client_to_run,
                            batch_size=client_cfg.batch_size,
                            local_steps=client_cfg.local_steps,
                            upload_size=ul_size,
                            download_size=dl_size,
                        )
                        self.round_evaluator.record_client(
                            client_to_run, dl_size, ul_size, exe_cost
                        )
                        logging.info(
                            f"Cannot prefetch, l {l} and r {r}   dl_size {dl_size}  prefetch_ratio {prefetched_ratio}  prefetch_size {prefetch_size} reason {'sticky or initial' if 0 <= r - l <= 1 else 'slow'}"
                        )

                roundDuration = (
                    exe_cost["computation"]
                    + exe_cost["downstream"]
                    + exe_cost["upstream"]
                )
                exe_cost["round"] = roundDuration
                virtual_client_clock[client_to_run] = exe_cost
                self.last_update_index[client_to_run] = (
                    self.round - 1
                )  # Client knows the global state from the previous round

                # if the client is not active by the time of collection, we consider it is lost in this round
                if self.client_manager.isClientActive(
                    client_to_run, roundDuration + self.global_virtual_clock
                ):
                    sampledClientsReal.append(client_to_run)
                    completionTimes.append(roundDuration)
                else:
                    sampledClientsLost.append(client_to_run)

            # raise Exception()
            num_clients_to_collect = min(num_clients_to_collect, len(completionTimes))
            # 2. get the top-k completions to remove stragglers
            workers_sorted_by_completion_time = sorted(
                range(len(completionTimes)), key=lambda k: completionTimes[k]
            )
            top_k_index = workers_sorted_by_completion_time[:num_clients_to_collect]
            clients_to_run = [sampledClientsReal[k] for k in top_k_index]

            dummy_clients = [
                sampledClientsReal[k]
                for k in workers_sorted_by_completion_time[num_clients_to_collect:]
            ]
            round_duration = completionTimes[top_k_index[-1]]
            completionTimes.sort()

            slowest_client_id = sampledClientsReal[top_k_index[-1]]
            logging.info(
                f"Successfully prefetch {len(prefetched_clients)} slowest client {slowest_client_id} is prefetched {slowest_client_id in prefetched_clients}  {completionTimes[-1]}"
            )
            # logging.info(f"Successfully prefetch {len(prefetched_clients)} slowest client {slowest_client_id} is prefetched {slowest_client_id in prefetched_clients}  {completionTimes}")

            return (
                clients_to_run,
                dummy_clients,
                sampledClientsLost,
                virtual_client_clock,
                round_duration,
                completionTimes[:num_clients_to_collect],
            )
        else:
            virtual_client_clock = {
                client: {"computation": 1, "communication": 1}
                for client in sampled_clients
            }
            completionTimes = [1 for c in sampled_clients]
            # return TictakResponse(sampled_clients, sampled_clients, [], virtual_client_clock, 1, compl)
            return (
                sampled_clients,
                sampled_clients,
                [],
                virtual_client_clock,
                1,
                completionTimes,
                -1,
            )

    def tictak_client_tasks(self, sampled_clients, num_clients_to_collect):
        """
        Record sampled client execution information in last round. In the SIMULATION_MODE,
        further filter the sampled_client and pick the top num_clients_to_collect clients.

        Args:
            sampled_clients (list of int): Sampled clients from client manager
            num_clients_to_collect (int): The number of clients actually needed for next round.

        Returns:
            tuple: Return the sampled clients and client execution information in the last round.

        """

        if len(sampled_clients) == 0:
            return [], [], [], {}, 0, []

        if self.experiment_mode == commons.SIMULATION_MODE:
            # NOTE: We try to remove dummy events as much as possible in simulations,
            # by removing the stragglers/offline clients in overcommitment"""
            sampledClientsReal = []
            sampledClientsLost = []
            completionTimes = []
            virtual_client_clock = {}

            # prefetch stats
            prefetched_clients = set()

            state_dict = self.model_wrapper.get_model().state_dict()
            layer_numels = []
            layer_element_sizes = [] # in bytes
            layer_sizes = [] # in bytes

            batchnorm_numel = 0 # number of batchnorm elements
            batchnorm_size = 0
            
            for key, tensor in state_dict.items():
                layer_numels.append(tensor.numel())
                layer_element_sizes.append(tensor.element_size())
                layer_sizes.append(tensor.numel() * tensor.element_size())
                if is_batch_norm_layer(key):
                    batchnorm_numel += tensor.numel()
                    batchnorm_size += tensor.numel() * tensor.element_size()
                # logging.info(f"Tensor {key}: numel {tensor.numel()} size {tensor.numel() * tensor.element_size()} element_size {tensor.element_size()} ")

            model_size = sum(layer_sizes) * 8 # bytes to bits
            model_numel = sum(layer_numels) # equivalent to the size of a bitmap

            batchnorm_size *= 8 # bytes to bits
            # logging.info(f"Total elem {model_numel}, batchnorm elem {batchnorm_numel}, batchnorm ratio {batchnorm_numel/model_numel}")

            # 1. remove dummy clients that are not available to the end of training
            for client_to_run in sampled_clients:
                client_cfg = self.client_conf.get(client_to_run, self.args)
                exe_cost = {
                    "computation": 0,
                    "downstream": 0,
                    "upstream": 0,
                    "round_duration": 0,
                }
                # =================================================
                dl_compressor = self.get_compressor(self.download_compressor_type)
                ul_compressor = self.get_compressor(self.upload_compressor_type)
                pre_compressor = self.get_compressor(self.prefetch_compressor_type)
                
                dl_numel = model_numel - batchnorm_numel
                ul_numel = model_numel - batchnorm_numel

                pre_numel = 0

                dl_aux_size = 0
                ul_aux_size = 0
                pre_aux_size = 0

                dl_size = model_size - batchnorm_size
                ul_size = model_size - batchnorm_size

                if self.args.compress_batch_norm:
                    # If batchnorm is compressed, then size/numels elements for download and upload equals the model's size/numel
                    dl_numel = model_numel
                    ul_numel = model_numel
                    dl_size = model_size
                    ul_size = model_size                    

                pre_size = 0

                l = self.last_update_index[client_to_run]
                l_pre = 0
                r = self.round - 1

                pre_ratio = 0

                if self.fl_method == FEDAVG:
                    dl_update_ratio = 1.0
                else:
                    dl_update_ratio = Sparsification.check_model_update_overhead(l, r, self.model_wrapper.get_model(), self.mask_record_list, self.device, use_accurate_cache=True)

                if self.enable_prefetch:                 
                    can_fully_prefetch = False

                    logging.info(f"Estimate prefetch client_id {client_to_run}, l {l}, r {r} and round {self.round}")

                    prefetch_start_i = (
                        1
                        if self.args.per_client_prefetch
                        else max(
                            min(self.max_prefetch_round + 1, self.round - 1) - 1, 1
                        )
                    )

                    for i in range(prefetch_start_i, min(self.max_prefetch_round + 1, self.round - 1)):
                        pl, pr = self.last_update_index[client_to_run], self.round - 1 - i
                        if pl >= pr:
                            logging.info(f"Unable to prefetch because client {client_to_run} participated recently")
                            break 

                        round_durations = self.round_evaluator.round_durations_aggregated[
                                max(0, self.round - 1 - i - self.max_prefetch_round) 
                                : self.round - 1 - i
                        ]
                        min_round_duration = min(round_durations)

                        prefetch_update_ratio = Sparsification.check_model_update_overhead(pl, pr, self.model_wrapper.get_model(), self.mask_record_list, self.device,  use_accurate_cache=True)

                        # Calculate the prefetch size
                        # TODO quantizing the prefetched model will reduce the model_size and is a case that need to be added
                        # An optimization, the shared mask will always be changed so there is no point in trying to transfer the model corresponding to the shared mask
                        pre_size = min(
                            model_size * (1 - self.shared_mask_ratio),
                            model_size * prefetch_update_ratio * (1 - self.shared_mask_ratio) + model_numel,
                        )
                        # Otherwise, we transfer the latest model or the gradient plus a bitmask
                        if self.shared_mask_ratio >= 1.0:
                            pre_size = min(
                            model_size,
                            model_size * prefetch_update_ratio + model_numel,
                        )

                        temp_pre_round = (
                            self.client_manager.get_download_time(client_to_run, pre_size)
                            / min_round_duration
                        )
                        logging.info(
                            f"Prefetch l {pl} r {pr} used min round duration {min_round_duration}, required prefetch round {temp_pre_round},  all usable round durations {round_durations} pre_size {pre_size}"
                        )

                        l_pre = self.round - 1 - i

                        # Represents how much of the prefetch_size can be downloaded in the prefetch window
                        # Note 0 <= pre_ratio <= 1, where pre_ratio = 1 means the client can fully prefetch the required amount within its window
                        pre_ratio = min(
                            sum(round_durations[-i:]) / self.client_manager.get_download_time(client_to_run, pre_size),
                            1.,
                        ) 

                        if temp_pre_round <= i:
                            can_fully_prefetch = True
                            break

                    if can_fully_prefetch:
                        dl_update_ratio = Sparsification.check_model_update_overhead(l_pre, r,  self.model_wrapper.get_model(), self.mask_record_list, self.device,  use_accurate_cache=True)
                        prefetched_clients.add(client_to_run)
                        logging.info(
                            f"After prefetch, l_pre {l_pre}, l {l} and r {r} prefetch_ratio {pre_ratio}  prefetch_size {pre_size}"
                        )
                        logging.info(f"dl_update_ratio {dl_update_ratio}")
                    elif l_pre > 0:
                        # Partial prefetch
                        # For case where the prefetch budget is not sufficient, but at least something has been fetched
                        # In this case, on start of the client's scheduled round:
                        # Finish prefetched portion gets an update equivalent to missing one round
                        # Unfinished portion gets an update equivalent to missing all the rounds since the client's last update
                        prefetched_dl_update_ratio = Sparsification.check_model_update_overhead(l_pre, r,  self.model_wrapper.get_model(), self.mask_record_list, self.device,  use_accurate_cache=True) * pre_ratio
                        unprefetched_dl_update_ratio = dl_update_ratio * (1. - pre_ratio)
                        dl_update_ratio = min(prefetched_dl_update_ratio + unprefetched_dl_update_ratio, 1.0)
                        logging.info(
                            f"Unable to fully prefetch, l_pre {l_pre}, l {l} and r {r} prefetch_ratio {pre_ratio}  prefetch_size {pre_size} reason {'sticky or initial' if 0 <= r - l <= 1 else 'slow'}"
                        )
                        logging.info(f"dl_update_ratio {dl_update_ratio} prefetched_dl_ratio {prefetched_dl_update_ratio} unprefetched_dl_ratio {unprefetched_dl_update_ratio}")

                    else:
                        # No prefetch case, only occurs at the start of training or when a client participated last round
                        pre_size = 0
                        pre_ratio = 0
                        logging.info(
                            f"Cannot prefetch, l_pre {l_pre}, l {l} and r {r}   dl_size {dl_size}  prefetch_ratio {pre_ratio}  prefetch_size {pre_size} reason {'sticky or initial' if 0 <= r - l <= 1 else 'slow'}"
                        )

                if self.fl_method == STC:
                    dl_numel = math.ceil(dl_numel * dl_update_ratio)
                    ul_numel = math.ceil(ul_numel * self.total_mask_ratio)

                    dl_aux_size += model_numel # Can actually be further reduced to just the number of non-batchnorm elements
                    ul_aux_size += model_numel

                    # TODO verify and remove
                    # If quantization is not used
                    dl_size = min(model_size * dl_update_ratio + model_numel, model_size)
                    ul_size = self.total_mask_ratio * model_size + model_numel
                elif self.fl_method == APF:
                    # TODO Verify and potentially remove
                    if self.last_update_index[client_to_run] == 0:
                        dl_update_ratio = 1.

                    dl_numel = math.ceil(dl_numel * dl_update_ratio) # Verify this case for prefetch
                    ul_numel = math.ceil(ul_numel * self.apf_ratio)

                    dl_aux_size += model_numel
                    ul_aux_size += 0

                    # TODO verify and remove
                    # If quantization is not used
                    dl_size = min(model_size * dl_update_ratio + model_numel, model_size)
                    ul_size = self.apf_ratio * model_size

                elif self.fl_method == GLUEFL:
                    dl_numel = math.ceil(dl_numel * dl_update_ratio)
                    ul_numel = math.ceil(ul_numel * self.total_mask_ratio)

                    dl_aux_size += model_numel
                    # Either a bitmap or indices of unique mask positions
                    ul_aux_size += min(model_numel, (self.total_mask_ratio - self.shared_mask_ratio) * model_numel * 32)

                    # TODO verify and remove
                    # If quantization is not used 
                    dl_size = min(model_size * dl_update_ratio + model_numel, model_size)
                    ul_size = self.total_mask_ratio * model_size + min((self.total_mask_ratio - self.shared_mask_ratio) * model_size, model_numel)


                # Apply quantization if enabled
                if self.download_compressor_type != NO_QUANTIZATION:
                    dl_size = dl_compressor.calculate_size(dl_numel) + dl_aux_size
                else:
                    # calculated_dl_size = dl_numel * 32 + batchnorm_size
                    # logging.info(f"dl_numel {dl_numel} dl model size {model_size} calculated model size {calculated_dl_size}")
                    pass
                
                if self.upload_compressor_type != NO_QUANTIZATION:
                    ul_size = ul_compressor.calculate_size(ul_numel) + ul_aux_size
                else:
                    pass

                
                if self.enable_prefetch and self.prefetch_compressor_type != NO_QUANTIZATION:
                    # TODO Quantization on prefetch is not implemented so this will never happen
                    pre_size = pre_compressor.calculate_size(pre_numel) + pre_aux_size

                if not self.args.compress_batch_norm:
                    dl_size += batchnorm_size
                    ul_size += batchnorm_size

                # Revert to sending the full model if the compressed download or upload size exceeds the full model size
                dl_size = min(dl_size, model_size)
                ul_size = min(ul_size, model_size)
                # No restrictions on prefetch size even though competetive algorithms shoul try to reduce the prefetch size by as much as possible

                exe_cost = self.client_manager.get_completion_time(client_to_run, batch_size=client_cfg.batch_size, local_steps=client_cfg.local_steps, upload_size=ul_size, download_size=dl_size)
                self.round_evaluator.record_client(client_to_run, dl_size, ul_size, exe_cost, prefetch_dl_size=pre_size)


                # =================================================

                
                roundDuration = (
                    exe_cost["computation"]
                    + exe_cost["downstream"]
                    + exe_cost["upstream"]
                )
                exe_cost["round"] = roundDuration
                virtual_client_clock[client_to_run] = exe_cost
                self.last_update_index[client_to_run] = (
                    self.round - 1
                )  # Client knows the global state from the previous round

                # if the client is not active by the time of collection, we consider it is lost in this round
                if self.client_manager.isClientActive(
                    client_to_run, roundDuration + self.global_virtual_clock
                ):
                    sampledClientsReal.append(client_to_run)
                    completionTimes.append(roundDuration)
                else:
                    sampledClientsLost.append(client_to_run)
            num_clients_to_collect = min(num_clients_to_collect, len(completionTimes))
            
            # 2. get the top-k completions to remove stragglers
            workers_sorted_by_completion_time = sorted(
                range(len(completionTimes)), key=lambda k: completionTimes[k]
            )
            top_k_index = workers_sorted_by_completion_time[:num_clients_to_collect]
            clients_to_run = [sampledClientsReal[k] for k in top_k_index]

            dummy_clients = [
                sampledClientsReal[k]
                for k in workers_sorted_by_completion_time[num_clients_to_collect:]
            ]
            round_duration = completionTimes[top_k_index[-1]]
            completionTimes.sort()

            slowest_client_id = sampledClientsReal[top_k_index[-1]]
            logging.info(
                f"Successfully prefetch {len(prefetched_clients)} slowest client {slowest_client_id} is prefetched {slowest_client_id in prefetched_clients}  {completionTimes[-1]}"
            )
            # logging.info(f"Successfully prefetch {len(prefetched_clients)} slowest client {slowest_client_id} is prefetched {slowest_client_id in prefetched_clients}  {completionTimes}")

            return (
                clients_to_run,
                dummy_clients,
                sampledClientsLost,
                virtual_client_clock,
                round_duration,
                completionTimes[:num_clients_to_collect],
            )
        else:
            virtual_client_clock = {
                client: {"computation": 1, "communication": 1}
                for client in sampled_clients
            }
            completionTimes = [1 for c in sampled_clients]
            # return TictakResponse(sampled_clients, sampled_clients, [], virtual_client_clock, 1, compl)
            return (
                sampled_clients,
                sampled_clients,
                [],
                virtual_client_clock,
                1,
                completionTimes,
                -1,
            )



    def client_completion_handler(self, results):
        """We may need to keep all updates from clients,
        if so, we need to append results to the cache

        Args:
            results (dictionary): client's training result

        """
        # Format:
        #       -results = {'client_id':client_id, 'update_weight': model_param, 'update_gradient': gradient_param, 'moving_loss': round_train_loss,
        #       'trained_size': count, 'wall_duration': time_cost, 'success': is_success 'utility': utility}

        if self.args.gradient_policy in ["q-fedavg"]:
            self.client_training_results.append(results)
        # Feed metrics to client sampler
        self.stats_util_accumulator.append(results["utility"])
        self.loss_accumulator.append(results["moving_loss"])

        self.client_manager.register_feedback(
            results["client_id"],
            results["utility"],
            auxi=math.sqrt(results["moving_loss"]),
            time_stamp=self.round,
            duration=self.virtual_client_clock[results["client_id"]]["computation"]
            + self.virtual_client_clock[results["client_id"]]["communication"],
        )

        # ================== Aggregate weights ======================
        self.update_lock.acquire()

        self.model_in_update += 1
        self.update_weight_aggregation(results)
        
        self.update_lock.release()

    def update_weight_aggregation(self, results):
        """May aggregate client updates on the fly"""

        # ===== initialize compressed gradients =====
        if self._is_first_result_in_round():
            self.compressed_gradient = [
                torch.zeros_like(param.data)
                .to(device=self.device, dtype=torch.float32)
                for param in self.model_wrapper.get_model().state_dict().values()
            ]

        keys = list(self.model_wrapper.get_model().state_dict().keys())

        # Perform decompression if needed
        # For now, the compressed then decompressed gradient is transmitted
        # so there is no need for decompression
        update_gradient = [
            torch.from_numpy(param).to(device=self.device) for param in results["update_gradient"]
        ]

        """
        update_gradient = None
        if self.upload_compressor_type == NO_QUANTIZATION:
            update_gradient = [
                torch.from_numpy(param).to(device=self.device) for param in results["update_gradient"]
            ]
        elif self.upload_compressor_type == QSGD:
            compressor = QSGDCompressor(self.args.quantization_level)
            update_gradient = self.apply_decompressor1(compressor, results["update_gradient"], keys, to_device=True)
        elif self.upload_compressor_type == QSGD_BUCKET:
            compressor = QSGDBucketCompressor(self.args.quantization_level)
            update_gradient = self.apply_decompressor1(compressor, results["update_gradient"], keys, to_device=True)
        elif self.upload_compressor_type == LFL:
            compressor = LFLCompressor(self.args.quantization_level)
            update_gradient = self.apply_decompressor1(compressor, results["update_gradient"], keys, to_device=True)
        else:
            raise NotImplementedError(f"Upload compression method {self.download_compressor_type} is not implemented")
        """

        # Aggregate gradients with specific gradient weights
        gradient_weight = self.get_gradient_weight(results["client_id"])
        # logging.info(f"weight: {self.get_gradient_weight(results['client_id'])} {results['client_id']}")
        
        for idx, key in enumerate(keys):
            if is_batch_norm_layer(key):
                # Batch norm layer is not weighted
                self.compressed_gradient[idx] += update_gradient[idx] * (1.0 / self.tasks_round)
            else:
                self.compressed_gradient[idx] += update_gradient[idx] * gradient_weight

        """
        for idx, param in enumerate(self.model_wrapper.get_model().state_dict().values()):
            # Batch norm layer is not weighted
            if not (("num_batches_tracked" in keys[idx]) or ("running" in keys[idx])):
                self.compressed_gradient[idx] += (
                    torch.from_numpy(results["update_gradient"][idx]).to(
                        device=self.device
                    )
                    * gradient_weight
                )
            else:
                self.compressed_gradient[idx] += torch.from_numpy(
                    results["update_gradient"][idx]
                ).to(device=self.device) * (1.0 / self.tasks_round)
        """

        # All clients are done
        if self._is_last_result_in_round():
            self.apply_and_update_mask()
            spar_ratio = Sparsification.check_sparsification_ratio(
                [self.compressed_gradient]
            )
            mask_ratio = Sparsification.check_sparsification_ratio([self.shared_mask])
            logging.info(f"Gradients sparsification: {spar_ratio}")
            logging.info(f"Mask sparsification: {mask_ratio}")

            # ==== update global model =====
            model_state_dict = self.model_wrapper.get_model().state_dict()
            for idx, param in enumerate(model_state_dict.values()):
                param.data = (
                    param.data.to(device=self.device).to(dtype=torch.float32)
                    - self.compressed_gradient[idx]
                )
            
            self.model_wrapper.get_model().load_state_dict(model_state_dict)

            # TODO Uncomment below to allow saving models in wandb, though model)state)dict.values() is a tensor so needs to be converted in ndarrays first
            # self.model_weights = [param.numpy() for param in model_state_dict.values()]

            # For testing quantification scaling factor
            if self.round == 1:
                self.first_state_dict = model_state_dict
            else:
                compressor = QSGDCompressor(127)
                first_scaling_factors = []
                for first_tensor, cur_tensor in zip(self.first_state_dict.values(), model_state_dict.values()):
                    tmp = first_tensor - cur_tensor
                    tmp_compressed, ctx = compressor.compress(tmp)
                    first_scaling_factors.append(tmp_compressed[1].numpy())
                logging.info(f"multi-round avg scaling factor {numpy.mean(first_scaling_factors)} (r: {self.round - 1})")


            # ===== update mask list =====
            mask_list = []
            for p_idx, key in enumerate(self.model_wrapper.get_model().state_dict().keys()):
                mask = (self.compressed_gradient[p_idx] != 0).to(
                    device=torch.device("cpu")
                )
                mask_list.append(mask)

            self.mask_record_list.append(mask_list)

            # ==== update quantized update =====
            self.update_quantized_update()
        

    def get_gradient_weight(self, client_id):
        prob = 0
        if self.sampling_strategy == "STICKY":
            if self.round <= 1:
                prob = 1.0 / float(self.tasks_round)
            elif client_id in self.sampled_sticky_client_set:
                prob = (1.0 / float(self.feasible_client_count)) * (
                    1.0
                    / (
                        (float(self.tasks_round) - float(self.sticky_group_change_num))
                        / float(self.sticky_group_size)
                    )
                )
            else:
                prob = (1.0 / float(self.feasible_client_count)) * (
                    1.0
                    / (
                        float(self.sticky_group_change_num)
                        / (
                            float(self.feasible_client_count)
                            - float(self.sticky_group_size)
                        )
                    )
                )
        else:
            """
            [FedAvg] "Communication-Efficient Learning of Deep Networks from Decentralized Data".
            H. Brendan McMahan, Eider Moore, Daniel Ramage, Seth Hampson, Blaise Aguera y Arcas. AISTATS, 2017
            """
            # Importance of each update is 1/#_of_participants
            # importance = 1./self.tasks_round
            prob = 1.0 / self.tasks_round
        return prob

    def apply_and_update_mask(self):
        compressor_tot = TopKCompressor(compress_ratio=self.total_mask_ratio)
        compressor_shr = TopKCompressor(compress_ratio=self.shared_mask_ratio)

        keys = []
        for idx, key in enumerate(self.model_wrapper.get_model().state_dict()):
            keys.append(key)

        for idx, param in enumerate(self.model_wrapper.get_model().state_dict().values()):
            if ("num_batches_tracked" in keys[idx]) or ("running" in keys[idx]):
                continue

            # --- STC ---
            if (
                self.fl_method in ["STC", "STCPrefetch"]
                or self.round % self.regenerate_epoch == 1
            ):
                # local mask
                self.compressed_gradient[idx], ctx_tmp = compressor_tot.compress(
                    self.compressed_gradient[idx]
                )

                self.compressed_gradient[idx] = compressor_tot.decompress(
                    self.compressed_gradient[idx], ctx_tmp
                )
            else:
                # shared + local mask
                update_mask = self.compressed_gradient[idx].clone().detach()
                update_mask[self.shared_mask[idx] == True] = numpy.inf
                update_mask, ctx_tmp = compressor_tot.compress(update_mask)
                update_mask = compressor_tot.decompress(update_mask, ctx_tmp)
                update_mask = update_mask.to(torch.bool)
                self.compressed_gradient[idx][update_mask != True] = 0.0

        # --- update shared mask ---
        for idx, param in enumerate(self.model_wrapper.get_model().state_dict().values()):
            if ("num_batches_tracked" in keys[idx]) or ("running" in keys[idx]):
                continue

            determined_mask = self.compressed_gradient[idx].clone().detach()
            determined_mask, ctx_tmp = compressor_shr.compress(determined_mask)
            determined_mask = compressor_shr.decompress(determined_mask, ctx_tmp)
            self.shared_mask[idx] = determined_mask.to(torch.bool)

    def get_client_conf(self, client_id):
        """Training configurations that will be applied on clients,
        developers can further define personalized client config here.

        Args:
            client_id (int): The client id.

        Returns:
            dictionary: TorchClient training config.

        """
        conf = {
            "learning_rate": self.args.learning_rate,
            "download_compressor_type": self.download_compressor_type,
            "upload_compressor_type": self.download_compressor_type
        }
        return conf
    
    def create_client_task(self, executorId):
        """Issue a new client training task to specific executor

        Args:
            executorId (int): Executor Id.

        Returns:
            tuple: Training config for new task. (dictionary, PyTorch or TensorFlow module)

        """
        next_client_id = self.resource_manager.get_next_task(executorId)
        train_config = None
        if next_client_id != None:
            config = self.get_client_conf(next_client_id)
            train_config = {
                "client_id": next_client_id,
                "task_config": config,
                # TODO move following to be part of task_config
                "agg_weight": (
                    self.get_gradient_weight(next_client_id)
                    * float(self.feasible_client_count)
                ),
            }
        return train_config, self.get_train_update_virtual()
    
    def get_train_update_virtual(self):
        """
        Transfer the client-side model that already applies the quantized update 
        """
        
        if (self.download_compressor_type == NO_QUANTIZATION) or self.round == 1:
            return self.model_wrapper.get_weights()
        else:
            return self.quantized_update
        
    def get_compressor(self, compressor_type):
        if compressor_type == NO_QUANTIZATION:
            return IdentityCompressor()
        if compressor_type == QSGD:
            return QSGDCompressor(self.args.quantization_level)
        elif compressor_type == QSGD_BUCKET:
            return QSGDBucketCompressor(self.args.quantization_level)
        elif compressor_type == LFL:
            return LFLCompressor(self.args.quantization_level)
        else:
            raise NotImplementedError(f"Download compression method {self.download_compressor_type} is not implemented")

    def update_quantized_update(self):
        model_weights = self.model_wrapper.get_weights_torch()
        keys = self.model_wrapper.get_keys()
        compressor = self.get_compressor(self.download_compressor_type)

        # Quantization is applied
        if self.quantization_target == FULL_MODEL:
            compressed_update = self.apply_compressor(compressor, model_weights, keys)
            decompressed_update = self.apply_decompressor(compressor, compressed_update, keys)
            self.quantized_update = [param.cpu() for param in decompressed_update]
        elif self.quantization_target == DIFF_MODEL:
            weight_diff = []
            for weight, prev_weight in zip(model_weights, self.prev_model_weights):
                weight_diff.append(weight - prev_weight)
            compressed_update = self.apply_compressor(compressor, weight_diff, keys)
            decompressed_update = self.apply_decompressor(compressor, compressed_update, keys)

            logging.info(f"Cross-round weight diff {Sparsification.check_sparsification_ratio([weight_diff])}")

            res = []
            for idx, key in enumerate(keys):
                res.append(self.prev_model_weights[idx] + decompressed_update[idx])
            self.prev_model_weights = copy.deepcopy(model_weights) # deepcopy likely not necessary

            self.quantized_update = [param.numpy() for param in res]
        elif self.quantization_target == DIFF_ESTIMATE:            
            weight_diff = []
            for weight, est_weight in zip(model_weights, self.client_estimate_weights):
                weight_diff.append(weight - est_weight)

            compressed_update = self.apply_compressor(compressor, weight_diff, keys)
            decompressed_update = self.apply_decompressor(compressor, compressed_update, keys)

            for idx, key in enumerate(keys):
                self.client_estimate_weights[idx] += decompressed_update[idx]

            self.quantized_update = [param.numpy() for param in self.client_estimate_weights]

        else:
            raise NotImplementedError(f"Download compression method {self.download_compressor_type} is not implemented")
        
    
    def apply_compressor(self, compressor: Compressor, params, keys: List[str]):
        res = []
        for param, key in zip(params, keys):
            cur_compressed_update, ctx = None, None
            # TODO Verify fl batch norm paper to see if quantization affects anything
            if is_batch_norm_layer(key):
                cur_compressed_update, ctx = param, (param.shape, "batch_norm")
            else:
                cur_compressed_update, ctx = compressor.compress(param)

            res.append((cur_compressed_update, ctx))

        return res
    
    def apply_decompressor(self, compressor: Compressor, params, keys: List[str]) -> List[torch.Tensor]:
        res = []
        for (param, ctx), key in zip(params, keys):
            cur_decompressed_update = None
            # TODO Verify fl batch norm paper to see if quantization affects anything
            if is_batch_norm_layer(key):
                cur_decompressed_update = param
            else:
                cur_decompressed_update = compressor.decompress(param, ctx)

            res.append(cur_decompressed_update)
        return res    


    def CLIENT_PING(self, request, context):
        """Handle client ping requests

        Args:
            request (PingRequest): Ping request info from executor.

        Returns:
            ServerResponse: Server response to ping request

        """
        # NOTE: client_id = executor_id in deployment,
        # while multiple client_id may use the same executor_id (VMs) in simulations
        executor_id, client_id = request.executor_id, request.client_id
        response_data = response_msg = commons.DUMMY_RESPONSE

        if len(self.individual_client_events[executor_id]) == 0:
            # send dummy response
            current_event = commons.DUMMY_EVENT
            response_data = response_msg = commons.DUMMY_RESPONSE
        else:
            current_event = self.individual_client_events[executor_id].popleft()
            if current_event == commons.CLIENT_TRAIN:
                response_msg, response_data = self.create_client_task(executor_id)

                if response_msg is None:
                    current_event = commons.DUMMY_EVENT
                    if self.experiment_mode != commons.SIMULATION_MODE:
                        self.individual_client_events[executor_id].append(
                            commons.CLIENT_TRAIN
                        )
            elif current_event == commons.MODEL_TEST:
                response_msg, response_data = self.get_test_config(client_id)
            elif current_event == commons.UPDATE_MODEL:
                # Transfer the entire model weights instead of partial model weights in real-life
                response_data = self.model_wrapper.get_weights()
            elif current_event == commons.UPDATE_MASK:
                response_data = self.get_shared_mask()
            elif current_event == commons.SHUT_DOWN:
                response_msg = self.get_shutdown_config(executor_id)

        response_msg, response_data = self.serialize_response(
            response_msg
        ), self.serialize_response(response_data)
        # NOTE: in simulation mode, response data is pickle for faster (de)serialization
        response = job_api_pb2.ServerResponse(
            event=current_event, meta=response_msg, data=response_data
        )
        if current_event != commons.DUMMY_EVENT:
            logging.info(f"Issue EVENT ({current_event}) to EXECUTOR ({executor_id})")

        return response

    def select_participants(self):
        """
        Selects next round's participants depending on the sampling strategy

        If sampling_strategy == "STICKY", then use sticky sampling and, if possible, ensure that the number of sticky
        and change/new/non-sticky clients is the same as specified in the config args.
        Relevant args include num_participants, sticky_group_size, sticky_group_change_num, overcommitment, overcommit_weight

        Otherwise, use uniform sampling.
        """
        if self.sampling_strategy == "STICKY":
            if self.enable_prefetch:
                if self.args.presample_strategy == SIMPLE:
                    self.sampled_sticky_clients, self.sampled_changed_clients = (
                        self.client_manager.presample_sticky_simple(
                            self.round, self.global_virtual_clock
                        )
                    )
                elif self.args.presample_strategy == UNIFORM:
                    self.sampled_sticky_clients, self.sampled_changed_clients = (
                        self.client_manager.presample_sticky_uniform(
                            self.round, self.global_virtual_clock
                        )
                    )
                elif self.args.presample_strategy == SPEED_PROB:
                    self.sampled_sticky_clients, self.sampled_changed_clients = (
                        self.client_manager.presample_sticky_speed_prob(
                            self.round, self.global_virtual_clock
                        )
                    )
            else:
                self.sampled_sticky_clients, self.sampled_changed_clients = (
                    self.client_manager.select_participants_sticky(
                        cur_time=self.global_virtual_clock
                    )
                )

            self.sampled_sticky_client_set = set(self.sampled_sticky_clients)

            # Choose fastest online changed clients
            (
                change_to_run,
                change_stragglers,
                change_lost,
                change_virtual_client_clock,
                change_round_duration,
                change_flatten_client_duration,
            ) = self.tictak_client_tasks(
                self.sampled_changed_clients,
                (
                    self.args.sticky_group_change_num
                    if self.round > 1
                    else self.args.num_participants
                ),
            )
            logging.info(
                f"Selected change participants to run: {sorted(change_to_run)}\nchange stragglers: {sorted(change_stragglers)}\nchange lost: {sorted(change_lost)}"
            )

            # Randomly choose from online sticky clients
            sticky_to_run_count = (
                self.args.num_participants - self.args.sticky_group_change_num
            )
            (
                sticky_fast,
                sticky_slow,
                sticky_lost,
                sticky_virtual_client_clock,
                _,
                sticky_flatten_client_duration,
            ) = self.tictak_client_tasks(
                self.sampled_sticky_clients, sticky_to_run_count
            )
            all_sticky = sticky_fast + sticky_slow
            all_sticky.sort(key=lambda c: sticky_virtual_client_clock[c]["round"])
            faster_sticky_count = sum(
                1
                for c in all_sticky
                if sticky_virtual_client_clock[c]["round"] <= change_round_duration
            )

            if faster_sticky_count >= sticky_to_run_count:
                sticky_to_run = random.sample(
                    all_sticky[:faster_sticky_count], sticky_to_run_count
                )
            else:
                extra_sticky_clients = sticky_to_run_count - faster_sticky_count
                sticky_to_run = all_sticky[: faster_sticky_count + extra_sticky_clients]
                logging.info(
                    f"Sticky group has only {faster_sticky_count} clients that are faster than change group, fastest sticky clients will be used"
                )

            if (
                len(self.sampled_sticky_clients) > 0
            ):  # There are no sticky clients in round 1
                slowest_sticky_client_id = max(
                    sticky_to_run, key=lambda k: sticky_virtual_client_clock[k]["round"]
                )
                sticky_round_duration = sticky_virtual_client_clock[
                    slowest_sticky_client_id
                ]["round"]
            else:
                sticky_round_duration = 0

            sticky_ignored = [c for c in all_sticky if c not in sticky_to_run]

            logging.info(
                f"Selected sticky participants to run: {sorted(sticky_to_run)}\nunselected sticky participants: {sorted(sticky_ignored)}\nsticky lost: {sorted(sticky_lost)}"
            )

            # Combine sticky and changed clients together
            self.clients_to_run = sticky_to_run + change_to_run
            self.round_stragglers = sticky_ignored + change_stragglers
            self.round_lost_clients = sticky_lost + change_lost
            self.virtual_client_clock = {
                **sticky_virtual_client_clock,
                **change_virtual_client_clock,
            }
            self.round_duration = max(sticky_round_duration, change_round_duration)
            self.flatten_client_duration = numpy.array(
                sticky_flatten_client_duration + change_flatten_client_duration
            )
            self.clients_to_run.sort(
                key=lambda k: self.virtual_client_clock[k]["round"]
            )
            self.slowest_client_id = self.clients_to_run[-1]

            # Make sure that there are change_num number of new clients added each epoch
            if (
                self.round > 1
                and self.fl_method != "GlueFLPrefetchB"
                and self.fl_method != "GlueFLPrefetchC"
            ):
                self.client_manager.update_sticky_group(change_to_run)

        else:
            if self.enable_prefetch:
                self.sampled_participants = self.client_manager.presample(
                    self.round, self.global_virtual_clock
                )
            else:
                self.sampled_participants = sorted(
                    self.client_manager.select_participants(
                        round(self.args.num_participants * self.args.overcommitment),
                        cur_time=self.global_virtual_clock,
                    )
                )
            logging.info(f"Sampled clients: {sorted(self.sampled_participants)}")

            (
                self.clients_to_run,
                self.round_stragglers,
                self.round_lost_clients,
                self.virtual_client_clock,
                self.round_duration,
                self.flatten_client_duration,
            ) = self.tictak_client_tasks(
                self.sampled_participants, self.args.num_participants
            )
            self.slowest_client_id = self.clients_to_run[-1]
            self.flatten_client_duration = numpy.array(self.flatten_client_duration)

        logging.info(
            f"Selected participants to run: {sorted(self.clients_to_run)}\nstragglers: {sorted(self.round_stragglers)}\nlost: {sorted(self.round_lost_clients)}"
        )

    def round_completion_handler(self):
        """Triggered upon the round completion, it registers the last round execution info,
        broadcast new tasks for executors and select clients for next round.
        """
        self.global_virtual_clock += self.round_duration
        self.round += 1

        last_round_avg_util = sum(self.stats_util_accumulator) / max(
            1, len(self.stats_util_accumulator)
        )
        # assign avg reward to explored, but not ran workers
        for client_id in self.round_stragglers:
            self.client_manager.register_feedback(
                client_id,
                last_round_avg_util,
                time_stamp=self.round,
                # TODO switch to using both download and upload time when we eventually test Oort
                duration=self.virtual_client_clock[client_id]["computation"] + self.virtual_client_clock[client_id]["communication"], 
                success=False,
            )

        avg_loss = sum(self.loss_accumulator) / max(1, len(self.loss_accumulator))
        logging.info(
            f"Wall clock: {round(self.global_virtual_clock)} s, round: {self.round}, Planned participants: "
            + f"{len(self.sampled_participants)}, Succeed participants: {len(self.stats_util_accumulator)}, Training loss: {avg_loss}"
        )

        # dump round completion information to tensorboard
        if len(self.loss_accumulator):
            self.log_train_result(avg_loss)

        if self.round > 1:
            self.round_evaluator.record_round_completion(
                self.clients_to_run,
                self.round_stragglers + self.round_lost_clients,
                self.slowest_client_id,
            )
            self.round_evaluator.print_stats()
            self.round_evaluator.start_new_round()

        # Select next round's participants
        self.select_participants()

        # Issue requests to the resource manager; Tasks ordered by the completion time
        self.resource_manager.register_tasks(self.clients_to_run)
        self.tasks_round = len(self.clients_to_run)

        # Update executors and participants
        if self.experiment_mode == commons.SIMULATION_MODE:
            self.sampled_executors = list(self.individual_client_events.keys())
        else:
            self.sampled_executors = [str(c_id) for c_id in self.sampled_participants]

        self.model_in_update = 0
        self.test_result_accumulator = []
        self.stats_util_accumulator = []
        self.client_training_results = []
        self.loss_accumulator = []
        self.update_default_task_config()

        if self.round >= self.args.rounds:
            self.broadcast_aggregator_events(commons.SHUT_DOWN)
        elif self.round % self.args.eval_interval == 0 or self.round == 5:
            self.broadcast_aggregator_events(commons.UPDATE_MODEL)
            self.broadcast_aggregator_events(commons.UPDATE_MASK)
            self.broadcast_aggregator_events(commons.MODEL_TEST) # Issues a START_ROUND after testing completes
        else:
            self.broadcast_aggregator_events(commons.UPDATE_MODEL)
            self.broadcast_aggregator_events(commons.UPDATE_MASK)
            self.broadcast_aggregator_events(commons.START_ROUND)

    def event_monitor(self):
        """Activate event handler according to the received new message"""
        logging.info("Start monitoring events ...")

        while True:
            # Broadcast events to clients
            if len(self.broadcast_events_queue) > 0:
                current_event = self.broadcast_events_queue.popleft()

                if current_event in (
                    commons.UPDATE_MODEL,
                    commons.MODEL_TEST,
                    commons.UPDATE_MASK,
                ):
                    self.dispatch_client_events(current_event)

                elif current_event == commons.START_ROUND:
                    self.dispatch_client_events(commons.CLIENT_TRAIN)

                elif current_event == commons.SHUT_DOWN:
                    self.dispatch_client_events(commons.SHUT_DOWN)
                    break

            # Handle events queued on the aggregator
            elif len(self.server_events_queue) > 0:
                client_id, current_event, meta, data = self.server_events_queue.popleft()

                if current_event == commons.UPLOAD_MODEL:
                    self.client_completion_handler(self.deserialize_response(data))
                    if len(self.stats_util_accumulator) == self.tasks_round:
                        self.round_completion_handler()

                elif current_event == commons.MODEL_TEST:
                    self.testing_completion_handler(
                        client_id, self.deserialize_response(data)
                    )

                else:
                    logging.error(f"Event {current_event} is not defined")

            else:
                # execute every 100 ms
                time.sleep(0.1)


if __name__ == "__main__":
    aggregator = PrefetchAggregator(parser.args)
    aggregator.run()
