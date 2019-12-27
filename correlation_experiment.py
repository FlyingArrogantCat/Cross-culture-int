from new_model.experiment_module import EnergyExperiment


if __name__ == '__main__':
    exp = EnergyExperiment(demography_flag=False, depth_memory=20)

    exp.make_experiment(list_energies=[1, 10, 100, 1000, 1e4], n_steps=100)
