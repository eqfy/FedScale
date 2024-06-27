from collections import deque
import logging
import math
from typing import Dict, List

import numpy as np
from scipy.special import softmax
from examples.prefetch.client_metadata import PrefetchClientMetadata
from examples.prefetch.constants import GLUEFL
from fedscale.cloud.client_manager import ClientManager


class PrefetchClientManager(ClientManager):
    def __init__(self, mode, args, sample_seed=233):
        super().__init__(mode, args, sample_seed)
        self.client_metadata: Dict[str, PrefetchClientMetadata] = {}
        self.sticky_group = []

        # Configs
        # self.sample_num = round(self.args.num_participants * self.args.overcommitment)
        self.sample_num = self.args.num_participants * self.args.overcommitment
        self.sticky_group_size = args.sticky_group_size  # aka "k"
        self.overcommitment = args.overcommitment
        numOfClientsOvercommit = round(
            args.num_participants * (args.overcommitment - 1.0)
        )
        if args.overcommit_weight >= 0:
            self.change_num = round(
                args.overcommit_weight * numOfClientsOvercommit
                + args.sticky_group_change_num
            )
        else:
            self.change_num = round(args.sticky_group_change_num * args.overcommitment)

        # Scheduling related
        self.max_prefetch_round = args.max_prefetch_round
        self.prefetch_estimation_start = args.prefetch_estimation_start
        self.overcommitment = args.overcommitment
        self.sampled_sticky_clients = deque()
        self.sampled_changed_clients = deque()
        self.sampled_clients = deque()
        self.historical_sticky_group = []

    # TODO Function to dynamically adjust extra_commit, may not be used in future designs.
    def init_sample_num(self) -> int:
        extra_commit = 1.3
        if not self.args.enable_prefetch:
            return self.args.num_participants * self.args.overcommitment
        

    def register_client(
        self,
        host_id: int,
        client_id: int,
        size: int,
        speed: Dict[str, float],
        duration: float = 1,
    ) -> None:
        """Register client information to the client manager.

        Args:
            hostId (int): executor Id.
            clientId (int): client Id.
            size (int): number of samples on this client.
            speed (Dict[str, float]): device speed (e.g., compuutation and communication).
            duration (float): execution latency.

        """
        uniqueId = self.getUniqueId(host_id, client_id)
        user_trace = (
            None
            if self.user_trace is None
            else self.user_trace[
                self.user_trace_keys[int(client_id) % len(self.user_trace)]
            ]
        )

        self.client_metadata[uniqueId] = PrefetchClientMetadata(
            host_id,
            client_id,
            speed,
            augmentation_factor=self.args.augmentation_factor,
            upload_factor=self.args.upload_factor,
            download_factor=self.args.download_factor,
            traces=user_trace,
        )

        # remove clients
        if size >= self.filter_less and size <= self.filter_more:
            self.feasibleClients.append(client_id)
            self.feasible_samples += size

            if self.mode == "oort":
                feedbacks = {'reward': min(size, self.args.local_steps * self.args.batch_size),
                             'duration': duration,
                             }
                self.ucb_sampler.register_client(client_id, feedbacks=feedbacks)
        else:
            del self.client_metadata[uniqueId]

    def get_round_sample_num(self) -> int:
        sample_num_floor = math.floor(self.sample_num)
        sample_num_ceil = math.ceil(self.sample_num)
        if sample_num_floor == sample_num_ceil:
            return sample_num_floor
        return sample_num_floor + int(self.rng.random() > (self.sample_num - sample_num_floor))

    def select_participants(self, cur_time: float = 0) -> List[int]:
        return super().select_participants(self.get_round_sample_num(), cur_time)

    # Sticky sampling
    def select_participants_sticky(self, cur_time):
        self.count += 1
        round_sample_num = self.get_round_sample_num()

        logging.info(
            f"Sticky sampling num {round_sample_num} K (sticky group size) {self.sticky_group_size} Change {self.change_num}"
        )
        clients_online = self.getFeasibleClients(cur_time)
        clients_online_set = set(clients_online)
        # logging.info(f"clients online: {clients_online}")
        if len(clients_online) <= round_sample_num:
            logging.error("Not enough online clients!")
            return clients_online, []

        selected_sticky_clients, selected_new_clients = [], []
        if len(self.sticky_group) == 0:
            # initalize the sticky group with overcommitment
            self.rng.shuffle(clients_online)
            client_len = round(
                min(self.sticky_group_size, len(clients_online) - 1)
                * self.overcommitment
            )
            temp_group = sorted(
                clients_online[:client_len], key=lambda c: min(self.getBwInfo(c))
            )
            self.sticky_group = temp_group[-self.sticky_group_size :]
            self.rng.shuffle(self.sticky_group)
            # We treat the clients sampled from the first round as sticky clients
            selected_new_clients = self.sticky_group[: min(round_sample_num, client_len)]
        else:
            # randomly delete some clients
            self.rng.shuffle(self.sticky_group)
            # Find the clients that are available in the sticky group
            online_sticky_group = [
                i for i in self.sticky_group if i in clients_online_set
            ]
            logging.info(f"num {round_sample_num} change {self.change_num}")
            selected_sticky_clients = online_sticky_group[
                : (round_sample_num - self.change_num)
            ]
            # randomly include some clients
            self.rng.shuffle(clients_online)
            client_len = min(self.change_num, len(clients_online) - 1)
            selected_new_clients = []
            for client in clients_online:
                if client in self.sticky_group:
                    continue
                selected_new_clients.append(client)
                if len(selected_new_clients) == client_len:
                    break

        logging.info(
            f"Selected sticky clients ({len(selected_sticky_clients)}): {sorted(selected_sticky_clients)}\nSelected new clients({len(selected_new_clients)}) {sorted(selected_new_clients)}"
        )
        return selected_sticky_clients, selected_new_clients

    def getBwInfo(self, clientId):
        return (
            self.client_metadata[self.getUniqueId(0, clientId)].dl_bandwidth,
            self.client_metadata[self.getUniqueId(0, clientId)].ul_bandwidth,
        )

    def update_sticky_group(self, new_clients):
        self.historical_sticky_group.append(self.sticky_group)
        self.rng.shuffle(self.sticky_group)
        self.sticky_group = self.sticky_group[: -len(new_clients)] + new_clients

    def get_completion_time(self, client_id, batch_size, local_steps, upload_size, download_size, in_bits=True):
        if in_bits: # convert to kbits
            upload_size = upload_size / 1024
            download_size = download_size / 1024
        return super().get_completion_time(client_id, batch_size, local_steps, upload_size, download_size)

    def get_download_time(self, clientId, size, in_bits=True):
        if in_bits:
            size = size / 1024
        return size / self.client_metadata[self.getUniqueId(0, clientId)].dl_bandwidth

    # TODO potentially move to separate scheduler class
    def presample_sticky_simple(self, round: int, cur_time: float, completed_clients=[]):
        if self.max_prefetch_round <= 0:
            return self.select_participants_sticky(cur_time)

        if round == 1:
            for _ in range(self.max_prefetch_round):
                sticky_client, changed_client = self.select_participants_sticky(
                    cur_time
                )
                self.sampled_sticky_clients.append(sticky_client)
                self.sampled_changed_clients.append(changed_client)
                # Sticky group is from a couple rounds before (self.max_prefetch_round)
                # Unfotunately, this also means that we are forfeiting some of the savings from sticky sampling

        sticky_client, changed_client = self.select_participants_sticky(cur_time)
        self.sampled_sticky_clients.append(sticky_client)
        self.sampled_changed_clients.append(changed_client)

        cur_sticky = self.sampled_sticky_clients.popleft()
        cur_change = self.sampled_changed_clients.popleft()

        # TODO validate the cur_sticky and cur_change groups
        # Try using more offline clien

        return cur_sticky, cur_change

    def presample_sticky_uniform(self, round: int, cur_time: float, completed_clients=[]):
        if self.max_prefetch_round <= 0:
            return self.select_participants_sticky(cur_time)

        if round == 1:
            for _ in range(self.max_prefetch_round):
                sticky_client, changed_client = self.select_participants_sticky(
                    cur_time
                )
                self.sampled_sticky_clients.append(sticky_client)
                self.sampled_changed_clients.append(changed_client)
                # Uniform
                new_changed_clients = self.rng.sample(
                    changed_client, min(self.change_num, len(changed_client))
                )
                self.update_sticky_group(new_changed_clients)

        sticky_client, changed_client = self.select_participants_sticky(cur_time)
        self.sampled_sticky_clients.append(sticky_client)
        self.sampled_changed_clients.append(changed_client)
        # Uniform
        new_changed_clients = self.rng.sample(
            changed_client, min(self.change_num, len(changed_client))
        )
        self.update_sticky_group(new_changed_clients)

        cur_sticky = self.sampled_sticky_clients.popleft()
        cur_change = self.sampled_changed_clients.popleft()

        # TODO validate the cur_sticky and cur_change groups
        # Try using more offline clien

        return cur_sticky, cur_change

    def presample_sticky_speed_prob(self, round: int, cur_time: float, completed_clients=[]):
        if self.max_prefetch_round <= 0:
            return self.select_participants_sticky(cur_time)

        if round == 1:
            for _ in range(self.max_prefetch_round):
                sticky_client, changed_client = self.select_participants_sticky(
                    cur_time
                )
                self.sampled_sticky_clients.append(sticky_client)
                self.sampled_changed_clients.append(changed_client)
                # A simple heuristic to sample the changed_client for next round
                # based on the idea that faster clients will be more frequently chosen
                changed_client_probability = softmax(
                    [min(self.getBwInfo(c)) for c in changed_client]
                )
                new_changed_clients = np.random.choice(
                    changed_client,
                    min(self.change_num, size=len(changed_client)),
                    replace=False,
                    p=changed_client_probability,
                ).tolist()
                self.update_sticky_group(new_changed_clients)

        sticky_client, changed_client = self.select_participants_sticky(cur_time)
        self.sampled_sticky_clients.append(sticky_client)
        self.sampled_changed_clients.append(changed_client)
        # A simple heuristic to sample the changed_client for next round
        # based on the idea that faster clients will be more frequently chosen
        changed_client_probability = softmax(
            [min(self.getBwInfo(c)) for c in changed_client]
        )
        new_changed_clients = np.random.choice(
            changed_client,
            min(self.change_num, size=len(changed_client)),
            replace=False,
            p=changed_client_probability,
        ).tolist()
        self.update_sticky_group(new_changed_clients)

        cur_sticky = self.sampled_sticky_clients.popleft()
        cur_change = self.sampled_changed_clients.popleft()

        # TODO validate the cur_sticky and cur_change groups
        # Try using more offline clien

        return cur_sticky, cur_change

    def presample(self, round: int, cur_time: float):
        if self.max_prefetch_round <= 0:
            return self.select_participants(cur_time=cur_time)

        if round == 1:
            for _ in range(self.max_prefetch_round):
                selected_clients = self.select_participants(cur_time=cur_time)
                self.sampled_clients.append(selected_clients)

        selected_clients = self.select_participants(cur_time=cur_time)
        self.sampled_clients.append(selected_clients)
        return self.sampled_clients.popleft()
    
    def get_replacement_client(self, cur_time: int, cur_round: int, avoid_list = []):
        sampling_frame = []
        if self.args.fl_method == GLUEFL:
            online_clients = set(self.getFeasibleClients(cur_time))
            sampling_frame = [i for i in self.historical_sticky_group[cur_round - 1] if (i in online_clients and i not in avoid_list)]
        else:
            sampling_frame = [i for i in self.getFeasibleClients(cur_time) if i not in avoid_list]
        return self.rng.choice(sampling_frame)
