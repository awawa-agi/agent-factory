import sys

def main():
    from .hparams import BasicFlowConfig

    command = sys.argv.pop(1) if len(sys.argv) > 1 else "help"
    
    if command == "basic_flow":
        config = BasicFlowConfig()

        from .training_flows import BasicFlow
        flow = BasicFlow(config)
        flow.run()

    else:
        print(f"Unknown command: {command}")

if __name__ == "__main__":
    main()