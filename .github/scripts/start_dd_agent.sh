#!/bin/bash

DD_API_KEY=$1
dataset=$2
config=$3

echo "Dataset: ${dataset}"
echo "Config: ${config}"

# Install Datadog system agent
DD_AGENT_MAJOR_VERSION=7 DD_API_KEY=$DD_API_KEY DD_SITE="datadoghq.eu" bash -c "$(curl -L https://s3.amazonaws.com/dd-agent/scripts/install_script.sh)"
DATADOG_YAML_PATH=/etc/datadog-agent/datadog.yaml
sudo chmod 666 $DATADOG_YAML_PATH

# Associate metrics with tags and env
{
    echo "env: rasa-regression-tests"
    echo "tags:"
    echo "- service:rasa"
    echo "- markusci:true"
    echo "- dataset:$dataset"
    echo "- config:$config"
} >> $DATADOG_YAML_PATH

# Enable system_core integration
sudo mv /etc/datadog-agent/conf.d/system_core.d/conf.yaml.example /etc/datadog-agent/conf.d/system_core.d/conf.yaml

# Apply changes
sudo service datadog-agent restart
