# evaluation.pyimport mir_eval
import mir_eval
import numpy as np
import librosa

def calculate_metrics(reference, estimated):
    sdr, sir, sar, _ = mir_eval.separation.bss_eval_sources(reference, estimated)
    return {"SDR": np.mean(sdr), "SIR": np.mean(sir), "SAR": np.mean(sar)}

def evaluate(reference_file, estimated_file):
    reference, _ = librosa.load(reference_file, sr=None)
    estimated, _ = librosa.load(estimated_file, sr=None)
    return calculate_metrics(reference, estimated)
    
if __name__ == "__main__":
    # Example reference and estimated signals
    reference = np.random.random((2, 10000))
    estimated = np.random.random((2, 10000))

    metrics = evaluate(reference, estimated)
    print(metrics)