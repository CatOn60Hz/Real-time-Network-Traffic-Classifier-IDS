import sys
import json
import pandas as pd
import joblib

def test_preprocessing():
    # Load preprocessor and features
    preprocessor = joblib.load('models/ids_preprocessor.pkl')
    original_features = joblib.load('models/original_feature_names.pkl')
    
    # Mock payload
    payload = {
        "dur": 0.12, "proto": "tcp", "service": "http", "state": "FIN",
        "spkts": 8, "dpkts": 20, 
        "sbytes": 800, "dbytes": 2000, "rate": 50,
        "sttl": 31, "dttl": 29, "sload": 400, "dload": 200,
        "sloss": 0, "dloss": 0, "sinpkt": 10, "dinpkt": 10,
        "sjit": 0, "djit": 0, "swin": 255, "stcpb": 1000, "dtcpb": 1000, "dwin": 255, 
        "tcprtt": 0.05, "synack": 0.02, "ackdat": 0.03, "smean": 64, "dmean": 64, 
        "trans_depth": 1, "response_body_len": 0, "ct_srv_src": 2, "ct_state_ttl": 1, 
        "ct_dst_ltm": 2, "ct_src_dport_ltm": 2, "ct_dst_sport_ltm": 1, "ct_dst_src_ltm": 2, 
        "is_ftp_login": 0, "ct_ftp_cmd": 0, "ct_flw_http_mthd": 1, "ct_src_ltm": 2, 
        "ct_srv_dst": 2, "is_sm_ips_ports": 0
    }

    input_df = pd.DataFrame([payload])
    
    for col in original_features:
        if col not in input_df.columns:
            if col in preprocessor.named_transformers_['cat'].named_steps['onehot'].feature_names_in_:
                input_df[col] = '-' 
            else:
                input_df[col] = 0 
                
        if col in preprocessor.named_transformers_['num'].named_steps['scaler'].feature_names_in_:
            input_df[col] = pd.to_numeric(input_df[col], errors='coerce').fillna(0)

    input_df = input_df[original_features]
    print("DataFrame shape:", input_df.shape)
    
    try:
        processed_features = preprocessor.transform(input_df)
        print("Transform successful. Output shape:", processed_features.shape)
    except Exception as e:
        print("Transform failed:", repr(e))

if __name__ == "__main__":
    test_preprocessing()
