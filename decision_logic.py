
def final_decision(cnn_prob, metadata_result):
    if "AI" in metadata_result:
        return "AI-Generated (Metadata Match)", 0.99
    elif cnn_prob > 0.8:
        return "AI-Generated (Visual Artifacts)", cnn_prob
    else:
        return "Likely Real", 1 - cnn_prob
