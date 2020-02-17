from model.model_with_nn_ver2.experiment_module import EnergyExperiment


if __name__ == '__main__':
    exp = EnergyExperiment(demography_flag=False, depth_memory=20)

    exp.make_experiment(list_energies=[0.1, 1, 10, 100, 1000, 1e4], n_steps=60)
