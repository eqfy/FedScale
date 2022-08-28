import os
import sys

import fedscale.core.channels.job_api_pb2 as job_api_pb2
from fedscale.core import commons
from fedscale.core.aggregation.aggregator import Aggregator
from fedscale.core.logger.aggragation import *
from fedscale.utils.compressor.topk import TopKCompressor


class FedDC_Aggregator(Aggregator):
    """Feed aggregator using tensorflow models"""
    def __init__(self, args):
        super().__init__(args)
        self.dataset_total_clients = args.dataset_total_clients # to distinguish between self.args.total_worker which is the total worker in a round
        self.num_participants = args.num_participants
        self.sticky_group_size = args.sticky_group_size
        self.change_num = args.change_num
        self.total_mask_ratio = args.total_mask_ratio  # = shared_mask + local_mask
        self.shared_mask_ratio = args.shared_mask_ratio
        self.regenerate_epoch = args.regenerate_epoch
        self.max_prefetch_round = args.max_prefetch_round
        self.sampling_strategy = args.sampling_strategy
        self.fl_method = args.fl_method

        self.compressed_gradient = None
        self.mask_record_list = []
        self.shared_mask = []

    def init_mask(self):
        self.shared_mask = []
        for idx, param in enumerate(self.model.state_dict().values()):
            self.shared_mask.append(torch.zeros_like(param, dtype=torch.bool).to(dtype=torch.bool))

    def run(self):
        """Start running the aggregator server by setting up execution 
        and communication environment, and monitoring the grpc message.
        """
        self.setup_env()
        self.init_control_communication()
        self.init_data_communication()

        self.init_model()
        self.init_mask()
        self.save_last_param()
        self.model_update_size = sys.getsizeof(
            pickle.dumps(self.model))/1024.0*8.  # kbits
        self.client_profiles = self.load_client_profile(
            file_path=self.args.device_conf_file)

        self.event_monitor()

    def get_shared_mask(self):
        """Get shared mask that would be used by all FL clients (in default FL)

        Returns:
            List of PyTorch tensor: Based on the executor's machine learning framework, initialize and return the mask for training.

        """
        return [p.to(device='cpu') for p in self.shared_mask]

    def client_completion_handler(self, results):
        """We may need to keep all updates from clients, 
        if so, we need to append results to the cache
        
        Args:
            results (dictionary): client's training result
        
        """
        # Format:
        #       -results = {'clientId':clientId, 'update_weight': model_param, 'update_gradient': gradient_param, 'moving_loss': round_train_loss,
        #       'trained_size': count, 'wall_duration': time_cost, 'success': is_success 'utility': utility}

        if self.args.gradient_policy in ['q-fedavg']:
            self.client_training_results.append(results)
        # Feed metrics to client sampler
        self.stats_util_accumulator.append(results['utility'])
        self.loss_accumulator.append(results['moving_loss'])

        self.client_manager.register_feedback(results['clientId'], results['utility'],
                                          auxi=math.sqrt(
                                              results['moving_loss']),
                                          time_stamp=self.round,
                                          duration=self.virtual_client_clock[results['clientId']]['computation'] +
                                          self.virtual_client_clock[results['clientId']
                                                                    ]['communication']
                                          )

        # ================== Aggregate weights ======================
        self.update_lock.acquire()

        self.model_in_update += 1
        self.aggregate_client_weights(results)

        # All clients are done
        if self.model_in_update == self.tasks_round:
            if self.fl_method == "APF":
                # TODO APF
                pass
            else:
                self.apply_and_update_mask()

            if self.fl_method == "APF":
                # TODO
                pass

            # ==== update global model =====
            model_state_dict = self.model.state_dict()
            for idx, param in enumerate(model_state_dict.values()):
                param.data = param.data.to(device=self.device).to(dtype=torch.float32) - self.compressed_gradient[idx]
            self.model.load_state_dict(model_state_dict)
            
            # ===== update mask list =====
            mask_list = []
            for p_idx, key in enumerate(self.model.state_dict().keys()):
                mask = (self.compressed_gradient[p_idx] != 0).to(self.device)
                mask_list.append(mask)

            self.mask_record_list.append(mask_list)
        self.update_lock.release()

    def aggregate_client_weights(self, results):
        """May aggregate client updates on the fly
        """

        # ===== initialize compressed gradients =====
        if self.model_in_update == 1:
            self.compressed_gradient = [torch.zeros_like(param.data).to(device=self.device).to(dtype=torch.float32) for param in self.model.state_dict().values()]

        prob = self.get_gradient_weight()
        for idx, param in enumerate(self.model.state_dict().values()):
            self.compressed_gradient[idx] += (torch.from_numpy(results['update_gradient'][idx]).to(device=self.device) * prob)

    def get_gradient_weight(self):
        prob = 0
        if self.sampling_strategy == "STICKY":
            #     if self.round <= 1:
            #         prob = (1.0 / float(self.tasks_round))
            #     elif client_is_sticky:
            #         prob = (1.0 / float(self.dataset_total_worker)) * (1.0 / ((float(self.tasks_round) - float(self.cur_change_num)) / float(self.tasks_round)))
            #     else:
            #         prob = (1.0 / float(self.dataset_total_worker)) * (1.0 / (float(self.cur_change_num) / (float(self.dataset_total_worker) - float(self.tasks_round))))
            #     # For debugging purposes, something is wrong if sticky_total_prob != 1 and all clients have finished
            #     self.sticky_total_prob += prob
            #     logging.info(f"client {results['clientId']} has prob {prob} round total prob {self.sticky_total_prob} is sticky {client_is_sticky}\n \
            #         round worker count {self.tasks_round}\t change_num {self.cur_change_num}\t data set {self.dataset_total_worker}")
            #     if self.model_in_update == self.tasks_round:
            #         self.sticky_total_prob = 0
            pass
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
        for idx, param in enumerate(self.model.state_dict().values()):
            # --- STC ---
            if self.fl_method == "STC" or self.round % self.regenerate_epoch == 1:
                # local mask
                self.compressed_gradient[idx], ctx_tmp = compressor_tot.compress(
                    self.compressed_gradient[idx])
                
                self.compressed_gradient[idx] = compressor_tot.decompress(self.compressed_gradient[idx], ctx_tmp)
            else:
                # shared + local mask
                max_value = float(self.compressed_gradient[idx].abs().max())
                update_mask = self.compressed_gradient[idx].clone().detach()
                update_mask[self.shared_mask[idx] == True] = max_value
                update_mask, ctx_tmp = compressor_tot.compress(update_mask)
                update_mask = compressor_tot.decompress(update_mask, ctx_tmp)
                update_mask = update_mask.to(torch.bool)
                self.compressed_gradient[idx][update_mask != True] = 0.0

        # --- update shared mask ---
        for idx, param in enumerate(self.model.state_dict().values()):
            determined_mask = self.compressed_gradient[idx].clone().detach()
            determined_mask, ctx_tmp = compressor_shr.compress(determined_mask)
            determined_mask = compressor_shr.decompress(determined_mask, ctx_tmp)
            self.shared_mask[idx] = determined_mask.to(torch.bool)

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
            current_event = self.individual_client_events[executor_id].popleft(
            )
            if current_event == commons.CLIENT_TRAIN:
                response_msg, response_data = self.create_client_task(
                    client_id)
                if response_msg is None:
                    current_event = commons.DUMMY_EVENT
                    if self.experiment_mode != commons.SIMULATION_MODE:
                        self.individual_client_events[executor_id].append(
                                commons.CLIENT_TRAIN)
            elif current_event == commons.MODEL_TEST:
                response_msg = self.get_test_config(client_id)
            elif current_event == commons.UPDATE_MODEL:
                response_data = self.get_global_model()
            elif current_event == commons.UPDATE_MASK:
                response_data = self.get_shared_mask()
            elif current_event == commons.SHUT_DOWN:
                response_msg = self.get_shutdown_config(executor_id)

        response_msg, response_data = self.serialize_response(
            response_msg), self.serialize_response(response_data)
        # NOTE: in simulation mode, response data is pickle for faster (de)serialization
        response = job_api_pb2.ServerResponse(event=current_event,
                                          meta=response_msg, data=response_data)
        if current_event != commons.DUMMY_EVENT:
            logging.info(f"Issue EVENT ({current_event}) to EXECUTOR ({executor_id})")
        
        return response

    def round_completion_handler(self):
        """Triggered upon the round completion, it registers the last round execution info,
        broadcast new tasks for executors and select clients for next round.
        """
        self.global_virtual_clock += self.round_duration
        self.round += 1

        if self.round % self.args.decay_round == 0:
            self.args.learning_rate = max(
                self.args.learning_rate*self.args.decay_factor, self.args.min_learning_rate)

        # handle the global update w/ current and last
        self.round_weight_handler(self.last_gradient_weights)

        avgUtilLastround = sum(self.stats_util_accumulator) / \
            max(1, len(self.stats_util_accumulator))
        # assign avg reward to explored, but not ran workers
        for clientId in self.round_stragglers:
            self.client_manager.register_feedback(clientId, avgUtilLastround,
                                              time_stamp=self.round,
                                              duration=self.virtual_client_clock[clientId]['computation'] +
                                              self.virtual_client_clock[clientId]['communication'],
                                              success=False)

        avg_loss = sum(self.loss_accumulator) / \
            max(1, len(self.loss_accumulator))
        logging.info(f"Wall clock: {round(self.global_virtual_clock)} s, round: {self.round}, Planned participants: " +
                     f"{len(self.sampled_participants)}, Succeed participants: {len(self.stats_util_accumulator)}, Training loss: {avg_loss}")

        # dump round completion information to tensorboard
        if len(self.loss_accumulator):
            self.log_train_result(avg_loss)

        # update select participants
        self.sampled_participants = self.select_participants(
            select_num_participants=self.args.num_participants, overcommitment=self.args.overcommitment)
        (clientsToRun, round_stragglers, virtual_client_clock, round_duration, flatten_client_duration) = self.tictak_client_tasks(
            self.sampled_participants, self.args.num_participants)

        logging.info(f"Selected participants to run: {clientsToRun}")

        # Issue requests to the resource manager; Tasks ordered by the completion time
        self.resource_manager.register_tasks(clientsToRun)
        self.tasks_round = len(clientsToRun)

        # Update executors and participants
        if self.experiment_mode == commons.SIMULATION_MODE:
            self.sampled_executors = list(
                self.individual_client_events.keys())
        else:
            self.sampled_executors = [str(c_id)
                                      for c_id in self.sampled_participants]

        self.save_last_param()
        self.round_stragglers = round_stragglers
        self.virtual_client_clock = virtual_client_clock
        self.flatten_client_duration = numpy.array(flatten_client_duration)
        self.round_duration = round_duration
        self.model_in_update = 0
        self.test_result_accumulator = []
        self.stats_util_accumulator = []
        self.client_training_results = []

        if self.round >= self.args.rounds:
            self.broadcast_aggregator_events(commons.SHUT_DOWN)
        elif self.round % self.args.eval_interval == 0:
            self.broadcast_aggregator_events(commons.UPDATE_MODEL)
            self.broadcast_aggregator_events(commons.UPDATE_MASK)
            self.broadcast_aggregator_events(commons.MODEL_TEST)
        else:
            self.broadcast_aggregator_events(commons.UPDATE_MODEL)
            self.broadcast_aggregator_events(commons.UPDATE_MASK)
            self.broadcast_aggregator_events(commons.START_ROUND)

    def event_monitor(self):
        """Activate event handler according to the received new message
        """
        logging.info("Start monitoring events ...")

        while True:
            # Broadcast events to clients
            if len(self.broadcast_events_queue) > 0:
                current_event = self.broadcast_events_queue.popleft()

                if current_event in (commons.UPDATE_MODEL, commons.MODEL_TEST, commons.UPDATE_MASK):
                    self.dispatch_client_events(current_event)

                elif current_event == commons.START_ROUND:
                    self.dispatch_client_events(commons.CLIENT_TRAIN)

                elif current_event == commons.SHUT_DOWN:
                    self.dispatch_client_events(commons.SHUT_DOWN)
                    break

            # Handle events queued on the aggregator
            elif len(self.sever_events_queue) > 0:
                client_id, current_event, meta, data = self.sever_events_queue.popleft()

                if current_event == commons.UPLOAD_MODEL:
                    self.client_completion_handler(
                        self.deserialize_response(data))
                    if len(self.stats_util_accumulator) == self.tasks_round:
                        self.round_completion_handler()

                elif current_event == commons.MODEL_TEST:
                    self.testing_completion_handler(
                        client_id, self.deserialize_response(data))

                else:
                    logging.error(f"Event {current_event} is not defined")

            else:
                # execute every 100 ms
                time.sleep(0.1)

if __name__ == "__main__":
    aggregator = FedDC_Aggregator(args)
    aggregator.run()