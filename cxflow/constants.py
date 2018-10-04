"""**cxflow** module containing various constants."""

CXF_LOG_FORMAT = '%(asctime)s.%(msecs)03d: %(levelname)-8s@%(module)-12s: %(message)s'
"""General logging format."""

CXF_LOG_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
"""Date format used in logging."""

CXF_FULL_DATE_FORMAT = '%Y-%m-%d %H:%M:%S.%f'
"""Full date format."""

CXF_HOOKS_MODULE = 'cxflow.hooks'
"""Module with standard cxflow hooks (as would be used in import)."""

CXF_CONFIG_FILE = 'config.yaml'
"""Configuration file name (dumped in the output directory)."""

CXF_DEFAULT_LOG_DIR = './log'
"""Default log directory."""

CXF_LOG_FILE = 'train.log'
"""Log file (dumped in the output directory)."""

CXF_TRACE_FILE = 'trace.yaml'
"""Training trace filename."""

CXF_DEFAULT_TRAIN_STREAM = 'train'
"""Train stream name."""

CXF_PREDICT_STREAM = 'predict'
"""Predict stream name."""

CXF_NA_STR = 'N/A'
"""N/A string for pretty printing."""

CXF_BUFFER_SLEEP = 0.02
"""The duration for which the buffer sleeps before it starts to process the next batch."""

__all__ = ['CXF_LOG_FORMAT', 'CXF_LOG_DATE_FORMAT', 'CXF_FULL_DATE_FORMAT', 'CXF_HOOKS_MODULE', 'CXF_CONFIG_FILE',
           'CXF_LOG_FILE', 'CXF_TRACE_FILE', 'CXF_TRAIN_STREAM', 'CXF_PREDICT_STREAM', 'CXF_DEFAULT_LOG_DIR',
           'CXF_NA_STR', 'CXF_BUFFER_SLEEP']
