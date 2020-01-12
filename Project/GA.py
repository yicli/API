import numpy as np
import random
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
# from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import scale
from preproc import unpick  # add /Project to pythonpath first
layers = tf.keras.layers
optimizer = tf.keras.optimizers
utils = tf.keras.utils

# set maximum CPU threads to 4
tf.config.threading.set_inter_op_parallelism_threads(2)
tf.config.threading.set_intra_op_parallelism_threads(2)


class Chromosome:
    def __init__(self, x, y, use_custom_mse=True):
        # experimental parameters as defined in the paper
        self.no_epochs = 5
        self.normal_init = tf.initializers.RandomNormal(mean=0, stddev=0.01)
        self.learning_rate = 0.001
        self.optimiser = optimizer.Adam(self.learning_rate)
        self.batch_size = 128
        self.units = 100
        self.rate = 0.8
        if tf.__version__[0] == '2':  # TF v2 define this differently
            self.rate = 1 - self.rate
        self.filters = 10
        self.kernel_size = (2, 2)
        self.activation = "relu"
        self.fitness = np.inf
        self.pool_size = (2, 2)
        self.stride = 1
        self.orig_x = x
        self.orig_y = y
        self.model = None
        self.use_custom_mse = use_custom_mse

        # chromosome construction
        self.genes = dict.fromkeys(("loss", "out_nodes", "activation", "architecture"))
        self.genes["loss"] = random.choice(("categorical_crossentropy", "mean_squared_error"))
        self.genes["out_nodes"] = random.choice((1, len(np.unique(y))))
        self.genes["activation"] = random.choice(("linear", "softmax", "relu", "sigmoid"))
        if len(self.orig_x.shape) == 4:
            self.genes["architecture"] = random.choices((layers.Conv2D, layers.Dense, layers.Dropout, layers.MaxPool2D),
                                                        k=random.randint(5, 15))
        elif len(self.orig_x.shape) == 2:
            self.genes["architecture"] = random.choices((layers.Dense, layers.Dropout),
                                                        k=random.randint(5, 15))
        else:
            raise Exception("unexpected input shape", self.orig_x.shape)

    @staticmethod
    def custom_mse_numpy(y, yhat):
        obj = tf.keras.losses.MeanSquaredError()
        if len(y.shape) == 1:
            return obj(y, yhat)
        elif len(y.shape) == 2:
            truth_col_ind = np.argmax(y, axis=1)
            seq_row_ind = range(len(y))
            return obj(y[seq_row_ind, truth_col_ind], yhat[seq_row_ind, truth_col_ind])
        else:
            raise Exception("unexpected y shape", y.shape)

    @staticmethod
    def custom_mse(mse_flag):
        if mse_flag:
            def custom_mse_loss(y, yhat):
                obj = tf.keras.losses.MeanSquaredError()
                return obj(y, yhat)
            return custom_mse_loss
        else:
            def custom_mse_loss(y, yhat):
                res = tf.multiply(y, yhat)
                res = tf.subtract(res, y)
                res = tf.multiply(res, res)
                res = tf.reduce_sum(res, axis=1)
                res = tf.reduce_mean(res)
                return res
            return custom_mse_loss

    def evaluate(self):
        self.fitness = np.inf
        # data process: one-hot encode y for crossentropy
        if self.genes["loss"] == "categorical_crossentropy":
            le_obj = LabelEncoder()
            mod_y = le_obj.fit_transform(self.orig_y)
            mod_y = utils.to_categorical(mod_y, len(np.unique(mod_y)))
        # scale y for MSE
        else:
            mod_y = scale(self.orig_y.reshape(-1, 1))
            mod_y = mod_y.reshape(-1)

        # build model
        self.model = tf.keras.models.Sequential()
        self.model.add(layers.Input(shape=np.shape(self.orig_x[0])))
        for layer in self.genes["architecture"]:
            try:
                if layer == layers.Conv2D:
                    self.model.add(layer(filters=self.filters,
                                         kernel_size=self.kernel_size,
                                         activation=self.activation,
                                         kernel_initializer=self.normal_init))
                elif layer == layers.Dense:
                    self.model.add(layer(units=self.units,
                                         activation=self.activation,
                                         kernel_initializer=self.normal_init))
                elif layer == layers.Dropout:
                    self.model.add(layer(rate=self.rate))
                else:
                    self.model.add(layer(pool_size=self.pool_size,
                                         strides=self.stride))
            except Exception as e:
                return  # exit when invalid config is encountered
        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(units=self.genes["out_nodes"],
                                    activation=self.genes["activation"]))

        # compile
        is_mse = self.genes["loss"] == "mean_squared_error"
        self.model.compile(loss=self.genes["loss"],
                           optimizer=self.optimiser,
                           metrics=["mean_squared_error", self.custom_mse(is_mse)])

        # end execution if the no. of output nodes is not appropriate for the loss function
        if self.genes["loss"] == "mean_squared_error" and self.genes["out_nodes"] != 1:
            return
        if self.genes["loss"] == "categorical_crossentropy" and self.genes["out_nodes"] == 1:
            return

        try:
            # compute validation loss on second half of data, in line with validation_split in the fit method
            halfway = self.orig_x.shape[0]
            halfway = int((halfway/2+1)//1)
            v0 = self.model.evaluate(self.orig_x[halfway:-1, ...], mod_y[halfway:-1, ...],
                                     verbose=0)
            self.model.fit(self.orig_x, mod_y,
                           batch_size=self.batch_size,
                           epochs=self.no_epochs,
                           verbose=0,
                           validation_split=0.5)  # validation data taken from tail of array before shuffling
        except RuntimeError as e:
            if str(e)[0:16] == "You must compile":  # models with invalid layer configs are not compiled during init
                return

        val_loss = self.model.history.history["val_loss"]
        if np.mean(val_loss) < v0[0]:  # check if learning has taken place
            if self.use_custom_mse:
                self.fitness = self.model.history.history["val_custom_mse_loss"][-1]
            else:
                self.fitness = self.model.history.history["val_mean_squared_error"][-1]


class GA:
    def __init__(self, x, y, pop_size=50, max_generations=10, co_rate=0.7, mut_rate=0.3, use_custom_mse=True):
        self.pop_size = pop_size
        self.max_generations = max_generations
        self.co_rate = co_rate
        self.mut_rate = mut_rate
        self.x = x
        self.y = y
        self.use_custom_mse = use_custom_mse
        self.hist = {"ce_fit": [], "mse_pct": [], "mse_fit": [], "best_mod": []}

    def update_hist(self, res_table, population):  # save per generation results
        mse_prop = np.sum(res_table[:, 0])/self.pop_size
        self.hist["mse_pct"].append(mse_prop)
        print("Proportion of MSE:", mse_prop)
        if mse_prop == 0:
            self.hist["mse_fit"].append(np.NaN)
            self.hist["ce_fit"].append(np.min(res_table[:, 1]))
        elif mse_prop == 1:
            self.hist["mse_fit"].append(np.min(res_table[:, 1]))
            self.hist["ce_fit"].append(np.NaN)
        else:
            mse_rows = res_table[:, 0] == 1
            self.hist["mse_fit"].append(np.min(res_table[mse_rows, 1]))
            self.hist["ce_fit"].append(np.min(res_table[~mse_rows, 1]))
        print("Best fitness MSE:", self.hist["mse_fit"][-1])
        print("Best fitness CE:", self.hist["ce_fit"][-1])
        self.hist["best_mod"].append(population[np.argmin(res_table[:, 1])].model)

    def select_parent(self, population, tournament_size=5):
        current_best = Chromosome(self.x, self.y, use_custom_mse=self.use_custom_mse)
        for i in range(tournament_size):
            random_chromosome = random.choice(population)
            if random_chromosome.fitness < current_best.fitness:
                current_best = random_chromosome
        return current_best

    @staticmethod
    def crossover(parent_1, parent_2):
        p = random.randint(0, len(parent_1.genes)-1)
        element = list(parent_1.genes)[p]
        temp = parent_1.genes[element]
        parent_1.genes[element] = parent_2.genes[element]
        parent_2.genes[element] = temp
        return parent_1, parent_2

    def mutate(self, parent):
        gene = random.choice(list(parent.genes))
        if gene == "loss":
            parent.genes["loss"] = random.choice(("categorical_crossentropy", "mean_squared_error"))
        elif gene == "out_nodes":
            parent.genes["out_nodes"] = random.choice((1, len(np.unique(self.y))))
        elif gene == "activation":
            parent.genes["activation"] = random.choice(("linear", "softmax", "relu", "sigmoid"))
        elif gene == "architecture":
            if len(self.x.shape) == 4:
                parent.genes["architecture"] = random.choices((layers.Conv2D, layers.Dense,
                                                               layers.Dropout, layers.MaxPool2D),
                                                              k=random.randint(5, 15))
            elif len(self.x.shape) == 2:
                parent.genes["architecture"] = random.choices((layers.Dense, layers.Dropout),
                                                              k=random.randint(5, 15))
            else:
                raise Exception("unexpected input shape", self.x.shape)
        return parent

    def run(self):
        # create initial population
        init_population = np.empty(self.pop_size, dtype=object)
        temp_res = np.zeros((self.pop_size, 2), dtype=np.float32)
        for i in range(self.pop_size):
            init_population[i] = Chromosome(self.x, self.y, use_custom_mse=self.use_custom_mse)
            init_population[i].evaluate()
            if init_population[i].genes["loss"] == "mean_squared_error":
                temp_res[i, 0] = 1
            temp_res[i, 1] = init_population[i].fitness
        self.update_hist(temp_res, init_population)

        gen = 0
        best_chromosome = Chromosome(self.x, self.y, use_custom_mse=self.use_custom_mse)
        current_population = init_population

        # start the genetic loop
        while gen < self.max_generations:
            gen = gen + 1
            print("Generation", gen)
            new_pop = np.empty(self.pop_size, dtype=object)

            # crossover
            for i in range(0, self.pop_size, 2):
                parent_1 = self.select_parent(current_population)
                parent_2 = self.select_parent(current_population)
                if random.uniform(0, 1) <= self.co_rate:
                    new_pop[i], new_pop[i+1] = self.crossover(parent_1, parent_2)
                else:
                    new_pop[i], new_pop[i+1] = parent_1, parent_2

            # mutate
            for j in range(self.pop_size):
                if random.uniform(0, 1) <= self.mut_rate:
                    new_pop[j] = self.mutate(new_pop[j])
            current_population = new_pop

            # evaluate
            temp_res = np.zeros((self.pop_size, 2), dtype=np.float32)
            for k in range(self.pop_size):
                current_population[k].evaluate()
                if current_population[k].genes["loss"] == "mean_squared_error":
                    temp_res[k, 0] = 1
                temp_res[k, 1] = current_population[k].fitness
                if current_population[k].fitness <= best_chromosome.fitness:
                    best_chromosome = current_population[k]
            self.update_hist(temp_res, current_population)
        return best_chromosome


class RandomSearch:
    def __init__(self, x, y, n=500):
        self.n = n
        self.x = x
        self.y = y

    def run(self):
        best_fit = np.inf
        best_loss = None
        for i in range(self.n):
            if i % 50 == 0:
                print("Evaluating config number:", i+1)
            config = Chromosome(self.x, self.y)
            config.evaluate()
            if config.fitness < best_fit:
                best_fit = config.fitness
                best_loss = config.genes["loss"]
        return best_fit, best_loss


if __name__ == "__main__":
    # example usage
    xx, yy = unpick("boston.pkl")  # set /AML as working dir

    # run API (GA)
    small_ga = GA(xx, yy, pop_size=10, max_generations=5)
    small_ga.run()

    # run random search baseline
    small_rndm_srch = RandomSearch(xx, yy, n=20)
    small_rndm_srch.run()
