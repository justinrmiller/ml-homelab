import streamlit as st
import ray
import boto3
import pandas as pd

st.title("ML Homelab Dashboard")

# Ray cluster status
if st.button("Check Ray Cluster"):
    try:
        ray.init(address='ray://ray-head:10001')
        st.success(f"Connected to Ray cluster with {ray.cluster_resources()['CPU']} CPUs")
        ray.shutdown()
    except Exception as e:
        st.error(f"Failed to connect: {e}")

# S3 Storage
if st.button("List S3 Buckets"):
    s3 = boto3.client('s3', 
      endpoint_url='http://localstack:4566',
      aws_access_key_id='test-key',
      aws_secret_access_key='test-secret'
    )
    try:
        buckets = s3.list_buckets()
        st.write(f"Buckets:\n{buckets['Buckets']}")
    except Exception as e:
        st.error(f"Failed to list buckets: {e}")
