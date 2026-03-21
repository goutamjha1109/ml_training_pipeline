import argparse
import yaml
import json
from logger import logger


def get_train_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--params', type=str, required=True)
    parser.add_argument('--schema', type=str, required=True)
    return parser.parse_args()


def get_validate_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--params', type=str, required=True)
    return parser.parse_args()


def load_schema(schema_path):
    try:
        with open(schema_path, "r") as f:
            schema = json.load(f)
        logger.info("Schema loaded successfully")
        return schema
    except FileNotFoundError:
        logger.error(f"Schema file not found: {schema_path}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Schema file is not valid JSON: {e}")
        raise


def load_params(params_path):
    try:
        with open(params_path, "r") as f:
            params = yaml.safe_load(f)
        logger.info("Params loaded successfully")
        return params
    except FileNotFoundError:
        logger.error(f"Params file not found: {params_path}")
        raise
    except yaml.YAMLError as e:
        logger.error(f"Params file is not valid YAML: {e}")
