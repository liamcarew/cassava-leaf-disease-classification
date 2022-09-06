from tensorflow import config

def get_peak_gpu_mem_usage(gpu_devices):

  #get peak GPU RAM usage
  if gpu_devices:
    peak_gpu = config.experimental.get_memory_info('GPU:0')['peak']

  return peak_gpu