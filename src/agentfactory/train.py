from agentfactory.hparams import BasicFlowConfig

def main():
    config = BasicFlowConfig()

    from agentfactory.training_flows.basic_flow import BasicFlow

    flow = BasicFlow(config)
    flow.run()

if __name__ == "__main__":
    main()