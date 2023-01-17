import logging


class RoundEvaluator(object):
    def __init__(self) -> None:
        self.round_durations_aggregated = []
        self.total_duration = 0.
        self.total_duration_dl = 0.
        self.total_duration_ul = 0.
        self.total_duration_compute = 0.
        self.avg_duration_dl = 0.
        self.avg_duration_ul = 0.
        self.avg_duration_compute = 0.

        self.cur_duration_dl = 0.
        self.cur_duration_ul = 0.
        self.cur_duration_compute = 0.
        self.cur_avg_duration_dl = 0.
        self.cur_avg_duration_ul = 0.
        self.cur_avg_duration_compute = 0.
        
        self.bandwidths = []
        self.total_bw = 0.
        self.total_bw_dl = 0.
        self.total_bw_ul = 0.
        self.total_bw_prefetch = 0.
        self.total_bw_oc = 0. # Bandwidth recording just the extra downstream from overcommitment

        self.round = 0

        # Per round
        self.cur_client_count = 0
        self.cur_bandwidth_total = 0.
        self.cur_durations = {} # client_id : { "upstream", "downstream", "computation"}
        self.cur_used_bandwidths = {} # client_id : {"upstream", "downstream", "prefetch"}
        self.cur_clients = [] # Unused all clients in this round

    def start_new_round(self):
        self.cur_client_count = 0
        self.cur_bandwidth_total = 0.
        self.cur_durations = {}
        self.cur_used_bandwidths = {}

    def record_client(self, client_id, dl_size, ul_size, duration, prefetch_dl_size = 0):
        self.cur_durations[client_id] = duration
        self.cur_used_bandwidths[client_id] = {"upstream": ul_size, "downstream": dl_size, "prefetch": prefetch_dl_size}


    def record_round_completion(self, clients_to_run, dummy_clients, slowest_client_id):
        self.round += 1

        round_duration = self.cur_durations[slowest_client_id]
        self.cur_duration_dl = round_duration["downstream"]
        self.cur_duration_ul = round_duration["upstream"]
        self.cur_duration_compute = round_duration["computation"]

        self.total_duration_dl += self.cur_duration_dl
        self.total_duration_ul += self.cur_duration_ul
        self.total_duration_compute += self.cur_duration_compute

        self.avg_duration_dl = (self.avg_duration_dl * (self.round - 1)) / self.round + self.cur_duration_dl/ self.round
        self.avg_duration_ul = (self.avg_duration_ul * (self.round - 1)) / self.round + self.cur_duration_ul / self.round
        self.avg_duration_compute = (self.avg_duration_compute * (self.round - 1)) / self.round + self.cur_duration_compute / self.round

        round_duration_aggregated = self.cur_duration_dl + self.cur_duration_ul + self.cur_duration_compute
        self.round_durations_aggregated.append(round_duration_aggregated)
        self.total_duration += round_duration_aggregated
        
        avg_ul_duration, avg_dl_duration, avg_compute_duration = 0, 0, 0
        for id in clients_to_run:
            bw = self.cur_used_bandwidths[id]
            self.total_bw += bw["upstream"] + bw["downstream"] + bw["prefetch"]
            self.total_bw_ul += bw["upstream"]
            self.total_bw_dl += bw["downstream"]
            self.total_bw_prefetch += bw["prefetch"]

            avg_ul_duration += self.cur_durations[id]["upstream"]
            avg_dl_duration += self.cur_durations[id]["downstream"]
            avg_compute_duration += self.cur_durations[id]["computation"]
        
        avg_ul_duration /= len(clients_to_run)
        avg_dl_duration /= len(clients_to_run)
        avg_compute_duration /= len(clients_to_run)
        self.cur_avg_duration_dl = (self.cur_avg_duration_dl * (self.round - 1))/ self.round + (avg_dl_duration / self.round)
        self.cur_avg_duration_ul = (self.cur_avg_duration_ul * (self.round - 1))/ self.round + (avg_ul_duration / self.round)
        self.cur_avg_duration_compute = (self.cur_avg_duration_compute * (self.round - 1))/ self.round + (avg_compute_duration / self.round)

        for id in dummy_clients:
            bw = self.cur_used_bandwidths[id]
            self.total_bw_oc += bw["downstream"]
        return self.total_bw, self.total_duration

    def print_stats(self, version=1):
        if version == 0:
            logging.info(f"Cumulative bandwidth usage:\n \
            total (excluding overcommit): {self.total_bw:.0f} kbit\n \
            total (including overcommit): {self.total_bw + self.total_bw_oc:.0f} kbit\n \
            downstream: {self.total_bw_dl:.0f} kbit\tupstream: {self.total_bw_ul:.0f} kbit\tprefetch: {self.total_bw_prefetch:.0f} kbit\tovercommit: {self.total_bw_oc:.0f} kbit")
            logging.info(f"Cumulative round durations:\n \
            (wall clock time) total:\t{self.total_duration:.0f} s\n \
            total_dl:\t{self.total_duration_dl:.0f} s\t \
            total_ul:\t{self.total_duration_ul:.0f} s\t \
            total_compute:\t{self.total_duration_compute:.0f} s\n \
            avg_dl:\t{self.avg_duration_dl:.0f} s\t \
            avg_ul:\t{self.avg_duration_ul:.0f} s\t \
            avg_compute:\t{self.avg_duration_compute:.0f} s\n \
            client_avg_dl:\t{self.cur_avg_duration_dl:.0f} s\t \
            client_avg_ul:\t{self.cur_avg_duration_ul:.0f} s\t \
            client_avg_compute:\t{self.cur_avg_duration_compute:.0f} s \
            ")
        elif version == 1:
            logging.info(f"Summary Stats Round {self.round}")
            logging.info(f"""Bandwidth Stats in kbits
            bw_all({self.total_bw:.0f}) 
            bw_all_oc({self.total_bw + self.total_bw_oc:.0f}) 
            bw_all_dl({self.total_bw_dl + self.total_bw_oc + self.total_bw_prefetch:.0f}) 
            bw_dl({self.total_bw_dl:.0f}) bw_up({self.total_bw_ul:.0f}) bw_oc({self.total_bw_oc:.0f}) 
            bw_prefetch({self.total_bw_prefetch:.0f})""")
            logging.info(f"""Time Stats in s
            t_all({self.total_duration:.2f}) t_dl({self.total_duration_dl:.2f}) t_ul({self.total_duration_ul:.2f}) t_compute({self.total_duration_compute:.2f})
            t_avg_dl({self.avg_duration_dl:.2f}) t_avg_ul({self.avg_duration_ul:.2f}) t_avg_compute({self.avg_duration_compute:.2f})
            t_cur_dl({self.cur_duration_dl:.2f}) t_cur_ul({self.cur_duration_ul:.2f}) t_cur_compute({self.cur_duration_compute:.2f})
            t_cur_avg_dl({self.cur_avg_duration_dl:.2f}) t_cur_avg_ul({self.cur_avg_duration_ul:.2f}) t_cur_avg_compute({self.cur_avg_duration_compute:.2f})
            """)