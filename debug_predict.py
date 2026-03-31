import sys
import json
import logging
from app import app, load_artifacts

# Setup basic logging to see everything
logging.basicConfig(level=logging.DEBUG)

def test_predict():
    with app.app_context():
        load_artifacts()
        
        # Manually create the payload exactly as index.html does
        # Using the 'normal' profile
        uDur = 0.12
        uRate = 50
        uSbytes = 800
        uDbytes = 2000
        basettl1 = 31
        basettl2 = 29
        
        payload = {
            "dur": uDur, "proto": "tcp", "service": "http", "state": "FIN",
            "spkts": max(1, int(uSbytes / 100)), 
            "dpkts": max(0, int(uDbytes / 100)),
            "sbytes": uSbytes, "dbytes": uDbytes, "rate": uRate,
            "sttl": basettl1, "dttl": basettl2, "sload": uRate * 8, "dload": (uRate/2) * 8,
            "sloss": 0, "dloss": 0, "sinpkt": 10, "dinpkt": 10,
            "sjit": 0, "djit": 0, "swin": 255, "stcpb": 1000, "dtcpb": 1000, "dwin": 255, 
            "tcprtt": 0.05, "synack": 0.02, "ackdat": 0.03, "smean": 64, "dmean": 64, 
            "trans_depth": 1, "response_body_len": 0, "ct_srv_src": 2, "ct_state_ttl": 1, 
            "ct_dst_ltm": 2, "ct_src_dport_ltm": 2, "ct_dst_sport_ltm": 1, "ct_dst_src_ltm": 2, 
            "is_ftp_login": 0, "ct_ftp_cmd": 0, "ct_flw_http_mthd": 1, "ct_src_ltm": 2, 
            "ct_srv_dst": 2, "is_sm_ips_ports": 0
        }

        # Make a mock request to the endpoint
        with app.test_client() as client:
            response = client.post('/predict', json=payload)
            print("Status code:", response.status_code)
            print("Response:", response.json)

if __name__ == "__main__":
    test_predict()
