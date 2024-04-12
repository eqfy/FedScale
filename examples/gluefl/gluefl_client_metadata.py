import numpy as np

from fedscale.cloud.internal.client_metadata import ClientMetadata


class GlueflClientMetadata(ClientMetadata):
    def __init__(
        self,
        hostId,
        clientId,
        speed,
        augmentation_factor=3.0,
        upload_factor=1.0,
        download_factor=1.0,
        traces=None,
    ) -> None:
        super().__init__(hostId, clientId, speed, traces)
        self.dl_bandwidth = speed["dl_kbps"]
        self.ul_bandwidth = speed["ul_kbps"]
        self.augmentation_factor = augmentation_factor
        self.upload_factor = upload_factor
        self.download_factor = download_factor

    def getCompletionTime(self, batch_size, upload_step, upload_size, download_size):
        """
        Computation latency: compute_speed is the inference latency of models (ms/sample). As reproted in many papers,
                             backward-pass takes around 2x the latency, so we multiple it by 3x;
        Communication latency: communication latency = (pull + push)_update_size/bandwidth;
        """
        return {
            "computation": self.augmentation_factor
            * batch_size
            * upload_step
            * float(self.compute_speed)
            / 1000.0,
            "communication": (upload_size + download_size) / float(self.bandwidth),
            "downstream": download_size / (self.dl_bandwidth * self.download_factor),
            "upstream": upload_size / (self.ul_bandwidth * self.upload_factor),
        }

    def getCompletionTime_lognormal(
        self,
        batch_size,
        local_steps,
        upload_step,
        upload_size,
        download_size,
        mean_seconds_per_sample=0.005,
        tail_skew=0.6,
    ):
        """
        Computation latency: compute_speed is the inference latency of models (ms/sample). As reproted in many papers,
                             backward-pass takes around 2x the latency, so we multiple it by 3x;
        Communication latency: communication latency = (pull + push)_update_size/bandwidth;
        """
        device_speed = max(0.0001, np.random.lognormal(1, tail_skew, 1)[0])
        return {
            "computation": device_speed
            * mean_seconds_per_sample
            * batch_size
            * local_steps,
            "communication": (upload_size + download_size) / float(self.bandwidth),
            "downstream": download_size / (self.dl_bandwidth * self.download_factor),
            "upstream": upload_size / (self.ul_bandwidth * self.upload_factor),
        }
