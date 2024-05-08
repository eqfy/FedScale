from collections import deque
import os
import sys
import math
import pickle
import time

import numpy
import torch
import fedscale.cloud.channels.job_api_pb2 as job_api_pb2

from examples.gluefl.gluefl_client_manager import GlueflClientManager
from fedscale.cloud import commons
from fedscale.cloud.aggregation.aggregator import Aggregator
from fedscale.cloud.logger.aggregator_logging import *
from fedscale.utils.compressor.topk import TopKCompressor
from fedscale.utils.eval.round_evaluator import RoundEvaluator
from fedscale.utils.eval.sparsification import Sparsification


class GlueflAggregator(Aggregator):
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

        self.last_update_index = []
        self.round_lost_clients = []
        self.clients_to_run = []
        self.slowest_client_id = -1
        self.round_evaluator = RoundEvaluator()

        # TODO Extract scheduler logic
        self.max_prefetch_round = args.max_prefetch_round
        self.prefetch_estimation_start = args.prefetch_estimation_start
        self.sampled_clients = []
        self.sampled_sticky_clients = []
        self.sampled_changed_clients = []
        # logging.info("Good Init")

    def init_client_manager(self, args):
        """Initialize Gluefl client sampler

        Args:
            args (dictionary): Variable arguments for fedscale runtime config. defaults to the setup in arg_parser.py

        Returns:
            GlueflClientManager: The client manager class
        """

        # sample_mode: random or oort
        client_manager = GlueflClientManager(args.sample_mode, args=args)

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
        
        self.event_monitor()

    def get_shared_mask(self):
        """Get shared mask that would be used by all FL clients (in default FL)

        Returns:
            List of PyTorch tensor: Based on the executor's machine learning framework, initialize and return the mask for training.

        """
        return [p.to(device="cpu") for p in self.shared_mask]

    def tictak_client_tasks(self, sampled_clients, num_clients_to_collect):
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
                if self.fl_method == "FedAvg":
                    exe_cost = self.client_manager.get_completion_time(
                        client_to_run,
                        batch_size=client_cfg.batch_size,
                        local_steps=client_cfg.local_steps,
                        upload_size=self.model_update_size,
                        download_size=self.model_update_size,
                    )
                    self.round_evaluator.record_client(
                        client_to_run, self.model_update_size, self.model_update_size, exe_cost
                    )
                elif self.fl_method == "STC":
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
                    )
                    self.round_evaluator.record_client(
                        client_to_run, dl_size, ul_size, exe_cost
                    )
                elif self.fl_method == "GlueFL":
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
                    )
                    self.round_evaluator.record_client(
                        client_to_run, dl_size, ul_size, exe_cost
                    )
                elif self.fl_method in [
                    "GlueFLPrefetchA",
                    "GlueFLPrefetchB",
                    "GlueFLPrefetchC",
                    "STCPrefetch",
                ]:
                    # This is an estimate by the server
                    can_fully_prefetch = False
                    prefetch_completed_round = 0
                    # These are the actual result of the prefetch
                    # 0 if participated recently, 1 if can fully prefetch, else (0, 1) if can partially prefetch.
                    prefetch_size = 0
                    prefetched_ratio = 0

                    logging.info(f"Last update index size {len(self.last_update_index)}")
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
                                client_to_run, prefetch_size
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
        # download_ratio = check_model_update_overhead(0, self.round - 1, self.model_wrapper.get_model(), self.mask_record_list)
        # logging.info(f"Download sparsification: {results['client_id']} {download_ratio}")

        self.model_in_update += 1
        self.update_gradient_aggregation(results)

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
                # logging.info(f"weight: {idx} {self.compressed_gradient[idx]}")
                param.data = (
                    param.data.to(device=self.device).to(dtype=torch.float32)
                    - self.compressed_gradient[idx]
                )
            self.model_wrapper.get_model().load_state_dict(model_state_dict)

            # ===== update mask list =====
            mask_list = []
            for p_idx, key in enumerate(self.model_wrapper.get_model().state_dict().keys()):
                mask = (self.compressed_gradient[p_idx] != 0).to(
                    device=torch.device("cpu")
                )
                mask_list.append(mask)

            self.mask_record_list.append(mask_list)
        self.update_lock.release()

    def update_gradient_aggregation(self, results):
        """May aggregate client updates on the fly"""

        # ===== initialize compressed gradients =====
        if self._is_first_result_in_round():
            self.compressed_gradient = [
                torch.zeros_like(param.data)
                .to(device=self.device)
                .to(dtype=torch.float32)
                for param in self.model_wrapper.get_model().state_dict().values()
            ]

        gradient_weight = self.get_gradient_weight(results["client_id"])
        # gradient_weight = (1.0 / self.tasks_round)
        # logging.info(f"weight: {self.get_gradient_weight(results['client_id'])} {results['client_id']}")
        keys = []
        for idx, key in enumerate(self.model_wrapper.get_model().state_dict()):
            keys.append(key)
        for idx, param in enumerate(self.model_wrapper.get_model().state_dict().values()):
            # Batch norm layer is not weighted
            if not (("num_batches_tracked" in keys[idx]) or ("running" in keys[idx])):
                # TODO Potentially remove this conversion by directly returning a a tensor?
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
                "agg_weight": (
                    self.get_gradient_weight(next_client_id)
                    * float(self.feasible_client_count)
                ),
            }
        return train_config, self.model_wrapper.get_weights()

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
            if self.fl_method == "GlueFLPrefetchA":
                self.sampled_sticky_clients, self.sampled_changed_clients = (
                    self.client_manager.presample_sticky_a(
                        self.round, self.global_virtual_clock
                    )
                )
            elif self.fl_method == "GlueFLPrefetchB":
                self.sampled_sticky_clients, self.sampled_changed_clients = (
                    self.client_manager.presample_sticky_b(
                        self.round, self.global_virtual_clock
                    )
                )
            elif self.fl_method == "GlueFLPrefetchC":
                self.sampled_sticky_clients, self.sampled_changed_clients = (
                    self.client_manager.presample_sticky_c(
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
            if self.fl_method == "STCPrefetch":
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
                duration=self.virtual_client_clock[client_id]["computation"]
                + self.virtual_client_clock[client_id]["communication"],
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
        elif self.round % self.args.eval_interval == 0 or self.round == 2:
            self.broadcast_aggregator_events(commons.UPDATE_MODEL)
            self.broadcast_aggregator_events(commons.UPDATE_MASK)
            self.broadcast_aggregator_events(commons.MODEL_TEST)
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
    aggregator = GlueflAggregator(parser.args)
    aggregator.run()
