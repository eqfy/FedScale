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
        self.shared_mask = []

    def get_shared_mask(self):
        """Get shared mask that would be used by all FL clients (in default FL)

        Returns:
            List of PyTorch tensor: Based on the executor's machine learning framework, initialize and return the mask for training.

        """
        return self.shared_mask

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
            # ===== TODO apply and generate mask =====

            if self.fl_method == "APF":
                # TODO APF
                pass
            else:
                compressor_tot = TopKCompressor(compress_ratio=self.total_mask_ratio)
                compressor_shr = TopKCompressor(compress_ratio=self.shared_mask_ratio)
                for idx, param in enumerate(self.model.state_dict().values()):
                    # --- STC ---
                    if self.fl_method == "STC" or self.epoch % self.regenerate_epoch == 1:
                        self.compressed_gradient[idx], ctx_tmp = compressor_tot.compress(
                            self.compressed_gradient[idx])
                        
                        self.compressed_gradient[idx] = compressor_tot.decompress(self.compressed_gradient[idx], ctx_tmp)
                    else:
                        # shared + local mask
                        # self.compressed_gradient[idx][self.mask_model[idx] != True] = 0.0
                        max_value = float(self.compressed_gradient[idx].abs().max())
                        update_mask = self.compressed_gradient[idx].clone().detach()
                        update_mask[self.mask_model[idx] == True] = max_value
                        update_mask, ctx_tmp = compressor_tot.compress(update_mask)
                        update_mask = compressor_tot.decompress(update_mask, ctx_tmp)
                        update_mask = update_mask.to(torch.bool)

                        self.compressed_gradient[idx][update_mask != True] = 0.0

                # --- update shared mask ---
                for idx, param in enumerate(self.model.state_dict().values()):
                    # shared mask
                    determined_mask = self.compressed_gradient[idx].clone().detach()
                    determined_mask, ctx_tmp = compressor_shr.compress(determined_mask)
                    determined_mask = compressor_shr.decompress(determined_mask, ctx_tmp)

                    self.mask_model[idx] = determined_mask.to(torch.bool)

            if self.fl_method == "APF":
                # TODO
                # mask_ratio = check_sparsification_ratio([self.mask_model])
                # logging.info(f"Mask sparsification: {mask_ratio}")
                # self.apf_ratio = mask_ratio
                pass

            self.last_global_gradient = self.compressed_gradient
            self.last_compressed_gradient = self.compressed_gradient
            model_state_dict = self.model.state_dict()
            for idx, param in enumerate(self.model_state_dict.values()):
                param.data = param.data.to(device=self.device).to(dtype=torch.float32) - self.compressed_gradient[idx]
            self.model.load_state_dict(model_state_dict)
            
            # ===== update mask =====
            mask_list = []
            for p_idx, key in enumerate(self.model.state_dict().keys()):

                mask = (self.compressed_gradient[p_idx] != 0).to(self.device)
                # if self.epoch <= 11:
                #     mask = torch.zeros_like(mask).to(device)
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
            #     if self.epoch <= 1:
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