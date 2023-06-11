from EnvLoader import EnvLoader
from Utils import rangeFloat, rangeInt


class Config:
    _env_loader = EnvLoader()
    
    @staticmethod
    def get_step_seed():
        return Config._env_loader.get_env_as_int("step_seed")
    
    @staticmethod
    def get_rf_range_ntrees():
        return rangeInt(
                Config._env_loader.get_env_as_int("rf_min_ntrees"), 
                Config._env_loader.get_env_as_int("rf_max_ntrees"), 
                Config._env_loader.get_env_as_int("rf_step_ntrees")
            )  
    
    @staticmethod
    def get_rf_range_max_depth():
        return rangeInt(
                Config._env_loader.get_env_as_int("rf_min_max_depth"), 
                Config._env_loader.get_env_as_int("rf_max_max_depth"), 
                Config._env_loader.get_env_as_int("rf_step_max_depth")
            )
        
    @staticmethod
    def get_rf_step_min_rows():
        return rangeFloat(
            Config._env_loader.get_env_as_float("rf_min_min_rows"), 
            Config._env_loader.get_env_as_float("rf_max_min_rows"), 
            Config._env_loader.get_env_as_float("rf_step_min_rows")
        )
        
    @staticmethod
    def get_rf_sample_rate():
        return rangeInt(
            Config._env_loader.get_env_as_int("rf_min_sample_rate"), 
            Config._env_loader.get_env_as_int("rf_max_sample_rate"), 
            Config._env_loader.get_env_as_int("rf_step_sample_rate")
        )
    
    @staticmethod
    def get_rf_stopping_tolerance():
        return Config._env_loader.get_env_as_float("rf_stopping_tolerance")
    
    @staticmethod
    def get_rf_stopping_rounds():
        return Config._env_loader.get_env_as_int("rf_stopping_rounds")
        
    @staticmethod
    def get_dl_range_size_hidden():
        return rangeInt(
                Config._env_loader.get_env_as_int("dl_min_size_hidden"), 
                Config._env_loader.get_env_as_int("dl_max_size_hidden"), 
                Config._env_loader.get_env_as_int("dl_step_size_hidden")
            )
    
    @staticmethod
    def get_dl_range_hidden():
        return rangeInt(
                Config._env_loader.get_env_as_int("dl_min_hidden"), 
                Config._env_loader.get_env_as_int("dl_max_hidden"), 
                Config._env_loader.get_env_as_int("dl_step_hidden")
            )
    
    @staticmethod
    def get_dl_range_epochs():
        return rangeInt(
                Config._env_loader.get_env_as_int("dl_min_epochs"), 
                Config._env_loader.get_env_as_int("dl_max_epochs"), 
                Config._env_loader.get_env_as_int("dl_step_epochs")
            )
        
    @staticmethod
    def get_dl_range_mini_batch_size():
        return rangeInt(
                Config._env_loader.get_env_as_int("dl_min_mini_batch_size"), 
                Config._env_loader.get_env_as_int("dl_max_mini_batch_size"), 
                Config._env_loader.get_env_as_int("dl_step_mini_batch_size")
            )
        
    @staticmethod
    def get_dl_range_rate():
        return rangeFloat(
                Config._env_loader.get_env_as_float("dl_min_rate"), 
                Config._env_loader.get_env_as_float("dl_max_rate"), 
                Config._env_loader.get_env_as_float("dl_step_rate")
            )
    
    @staticmethod
    def get_dl_stopping_tolerance():
        return Config._env_loader.get_env_as_float("dl_stopping_tolerance")
    
    @staticmethod
    def get_dl_stopping_rounds():
        return Config._env_loader.get_env_as_int("dl_stopping_rounds")
       
    @staticmethod
    def get_gbm_range_ntrees():
        return rangeInt(
                Config._env_loader.get_env_as_int("gbm_min_ntrees"), 
                Config._env_loader.get_env_as_int("gbm_max_ntrees"), 
                Config._env_loader.get_env_as_int("gbm_step_ntrees")
            )  
    
    @staticmethod
    def get_gbm_range_max_depth():
        return rangeInt(
                Config._env_loader.get_env_as_int("gbm_min_max_depth"), 
                Config._env_loader.get_env_as_int("gbm_max_max_depth"), 
                Config._env_loader.get_env_as_int("gbm_step_max_depth")
            )
        
    @staticmethod
    def get_gbm_range_min_rows():
        return rangeFloat(
            Config._env_loader.get_env_as_float("gbm_min_min_rows"), 
            Config._env_loader.get_env_as_float("gbm_max_min_rows"), 
            Config._env_loader.get_env_as_float("gbm_step_min_rows")
        )
        
    @staticmethod
    def get_gbm_range_sample_rate():
        return rangeInt(
            Config._env_loader.get_env_as_int("gbm_min_sample_rate"), 
            Config._env_loader.get_env_as_int("gbm_max_sample_rate"), 
            Config._env_loader.get_env_as_int("gbm_step_sample_rate")
        )
            
    @staticmethod
    def get_gbm_range_col_sample_rate():
        return rangeInt(
            Config._env_loader.get_env_as_int("gbm_min_col_sample_rate"), 
            Config._env_loader.get_env_as_int("gbm_max_col_sample_rate"), 
            Config._env_loader.get_env_as_int("gbm_step_col_sample_rate")
        )

    @staticmethod
    def get_gbm_range_learn_rate():
        return rangeFloat(
            Config._env_loader.get_env_as_float("gbm_min_learn_rate"), 
            Config._env_loader.get_env_as_float("gbm_max_learn_rate"), 
            Config._env_loader.get_env_as_float("gbm_step_learn_rate")
        )
    
    @staticmethod
    def get_gbm_stopping_tolerance():
        return Config._env_loader.get_env_as_float("gbm_stopping_tolerance")
    
    @staticmethod
    def get_gbm_stopping_rounds():
        return Config._env_loader.get_env_as_int("gbm_stopping_rounds")
    
    @staticmethod
    def get_pop_size():
        return Config._env_loader.get_env_as_int("pop_size")
        
    @staticmethod
    def get_max_gen():
        return Config._env_loader.get_env_as_int("max_gen")
        
    @staticmethod
    def get_mutation_rate():
        return Config._env_loader.get_env_as_float("mutation_rate")
        
    @staticmethod
    def get_crossover_rate():
        return Config._env_loader.get_env_as_float("crossover_rate")
        
    @staticmethod
    def get_max_prec_crossover_replace_params_values():
        return Config._env_loader.get_env_as_float("max_prec_crossover_replace_params_values")
        
    @staticmethod
    def get_max_prec_mutation_replace_params_values():
        return Config._env_loader.get_env_as_float("max_prec_mutation_replace_params_values")
        
    @staticmethod
    def get_verbose():
        return Config._env_loader.get_env_as_bool("verbose")
        
    @staticmethod
    def get_select_random_percent():
        return Config._env_loader.get_env_as_float("select_random_percent")
        
    @staticmethod
    def get_fitness_mse_percent():
        return Config._env_loader.get_env_as_float("fitness_mse_percent")
    
    @staticmethod
    def get_fitness_rmse_percent():
        return Config._env_loader.get_env_as_float("fitness_rmse_percent")
    
    @staticmethod
    def get_fitness_logloss_percent():
        return Config._env_loader.get_env_as_float("fitness_logloss_percent")
    
    @staticmethod
    def get_fitness_mean_per_class_error_percent():
        return Config._env_loader.get_env_as_float("fitness_mean_per_class_error_percent")
    
    @staticmethod
    def get_fitness_r2_percent():
        return Config._env_loader.get_env_as_float("fitness_r2_percent")
    
    @staticmethod
    def get_fitness_accuracy_percent():
        return Config._env_loader.get_env_as_float("fitness_accuracy_percent")
    
    @staticmethod
    def get_train_ratio():
        return Config._env_loader.get_env_as_float("train_ratio")
    
    @staticmethod
    def get_test_ratio():
        return Config._env_loader.get_env_as_float("test_ratio")
    
    @staticmethod
    def get_data_percentage_use_in_ga():
        return Config._env_loader.get_env_as_float("data_percentage_use_in_ga")
    
    @staticmethod
    def get_data_percentage_use_in_ga():
        return Config._env_loader.get_env_as_float("data_percentage_use_in_ga")
    
    @staticmethod
    def get_n_threads_percentage():
        return Config._env_loader.get_env_as_float("n_threads_percentage")
    
    @staticmethod
    def get_ram_percentage():
        return Config._env_loader.get_env_as_float("ram_percentage")