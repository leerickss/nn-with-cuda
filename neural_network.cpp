#include "neural_network.h"

#include <armadillo>
#include <vector>
#include "utils/common.h"
#include "gpu_func.h"
#include "mpi.h"
#include "iomanip"

#define MPI_SAFE_CALL( call ) do {                               \
    int err = call;                                              \
    if (err != MPI_SUCCESS) {                                    \
        fprintf(stderr, "MPI error %d in file '%s' at line %i",  \
               err, __FILE__, __LINE__);                         \
        exit(1);                                                 \
    } } while(0)


struct gpu_cache {

    double *W0;
    double *W1;
    double *b0;
    double *b1;   

    double *X;
    double *X_t;
    double *z0;
    double *a0;
    double *z1;
    double *a1;
    double *yc;

    double *y;
    double *diff;
    double *a0_t;
    double *dW1;
    double *db1;
    double *da0;
    double *W1_t;
    double *dz0;
    double *dW0;
    double *db0;

    int W0_rows, W0_cols, W1_rows, W1_cols;
    int b0_rows, b0_cols, b1_rows, b1_cols; 

    int X_rows, X_cols;
    int X_t_rows, X_t_cols;
    int z0_rows, z0_cols;
    int a0_rows, a0_cols;
    int z1_rows, z1_cols;
    int a1_rows, a1_cols;
    int yc_rows, yc_cols;

    int y_rows, y_cols;
    int diff_rows, diff_cols;
    int a0_t_rows, a0_t_cols;
    int dW1_rows, dW1_cols;
    int db1_rows, db1_cols;
    int da0_rows, da0_cols;
    int W1_t_rows, W1_t_cols;
    int dz0_rows, dz0_cols;
    int dW0_rows, dW0_cols;
    int db0_rows, db0_cols;

    //ctor
    gpu_cache(NeuralNetwork& nn, int batch_size) {
        init_dims(nn, batch_size);
        init_device_alloc();
    } 
    
    //dtor
    ~gpu_cache(){
        cudaFree(W0);
        cudaFree(W1);
        cudaFree(b0);
        cudaFree(b1);
        cudaFree(X);
        cudaFree(X_t);
        cudaFree(z0);
        cudaFree(a0);
        cudaFree(z1);
        cudaFree(a1);
        cudaFree(yc);
        cudaFree(y);
        cudaFree(diff);
        cudaFree(a0_t);
        cudaFree(dW1);
        cudaFree(db1);
        cudaFree(da0);
        cudaFree(W1_t);
        cudaFree(dz0);
        cudaFree(dW0);
        cudaFree(db0);
    }

    void init_dims(NeuralNetwork& nn, int batch_size) {
         
        W0_rows = nn.W[0].n_rows;
        W0_cols = nn.W[0].n_cols; 
        W1_rows = nn.W[1].n_rows;
        W1_cols = nn.W[1].n_cols;
        b0_rows = nn.b[0].n_rows;
        b0_cols = nn.b[0].n_cols;
        b1_rows = nn.b[1].n_rows;
        b1_cols = nn.b[1].n_cols;

        X_rows   = W0_cols;
        X_cols   = batch_size;
        X_t_rows = batch_size;
        X_t_cols = W0_cols;
        z0_rows  = nn.b[0].n_rows;
        z0_cols  = batch_size;
        a0_rows  = nn.b[0].n_rows;
        a0_cols  = batch_size;
        z1_rows  = nn.b[1].n_rows; 
        z1_cols  = batch_size;
        a1_rows  = nn.b[1].n_rows;
        a1_cols  = batch_size;
        yc_rows  = nn.b[1].n_rows; 
        yc_cols  = batch_size;

        y_rows    = W1_rows;
        y_cols    = batch_size;
        diff_rows = W1_rows;
        diff_cols = batch_size;
        a0_t_rows = batch_size;
        a0_t_cols = nn.b[0].n_rows;
        dW1_rows  = nn.W[1].n_rows;
        dW1_cols  = nn.W[1].n_cols;
        db1_rows  = nn.b[1].n_rows;
        db1_cols  = nn.b[1].n_cols;
        da0_rows  = nn.b[0].n_rows;
        da0_cols  = batch_size;
        W1_t_rows = nn.W[1].n_cols;
        W1_t_cols = nn.W[1].n_rows;
        dz0_rows  = nn.b[0].n_rows;
        dz0_cols  = batch_size;  
        dW0_rows  = nn.W[0].n_rows;
        dW0_cols  = nn.W[0].n_cols;
        db0_rows  = nn.b[0].n_rows;
        db0_cols  = nn.b[0].n_cols;
    }

    void update_dims(int batch_size){
        X_cols   = batch_size; 
        X_t_rows = batch_size;
        z0_cols  = batch_size;
        a0_cols  = batch_size;
        z1_cols  = batch_size;
        a1_cols  = batch_size;
        yc_cols  = batch_size;

        y_cols    = batch_size;
        diff_cols = batch_size;
        a0_t_rows = batch_size;
        da0_cols  = batch_size;
        dz0_cols  = batch_size;
    }
    
    int size(int n_rows, int n_cols) {
        return n_rows * n_cols * sizeof(double);
    }

    void init_device_alloc() {
        cudaMalloc((double**) &W0,   size(W0_rows,  W0_cols));
        cudaMalloc((double**) &W1,   size(W1_rows,  W1_cols));
        cudaMalloc((double**) &b0,   size(b0_rows,  b0_cols));
        cudaMalloc((double**) &b1,   size(b1_rows,  b1_cols));
        cudaMalloc((double**) &X,    size(X_rows,   X_cols));
        cudaMalloc((double**) &X_t,  size(X_t_rows, X_t_cols));
        cudaMalloc((double**) &z0,   size(z0_rows,  z0_cols));
        cudaMalloc((double**) &a0,   size(a0_rows,  a0_cols));
        cudaMalloc((double**) &z1,   size(z1_rows,  z1_cols));
        cudaMalloc((double**) &a1,   size(a1_rows,  z1_cols));
        cudaMalloc((double**) &yc,   size(yc_rows,  yc_cols));
        cudaMalloc((double**) &y,    size(y_rows,   y_cols));
        cudaMalloc((double**) &diff, size(diff_rows, diff_cols));
        cudaMalloc((double**) &a0_t, size(a0_t_rows, a0_t_cols));
        cudaMalloc((double**) &dW1,  size(dW1_rows,  dW1_cols));
        cudaMalloc((double**) &db1,  size(db1_rows,  db1_cols));
        cudaMalloc((double**) &da0,  size(da0_rows,  da0_cols));
        cudaMalloc((double**) &W1_t, size(W1_t_rows, W1_t_cols));
        cudaMalloc((double**) &dz0,  size(dz0_rows,  dz0_cols));
        cudaMalloc((double**) &dW0,  size(dW0_rows,  dW0_cols));
        cudaMalloc((double**) &db0,  size(db0_rows,  db0_cols));
    }

    void updateDeviceFromHost(NeuralNetwork& nn) {
        cudaMemcpy(W0, nn.W[0].memptr(), size(W0_rows, W0_cols), cudaMemcpyHostToDevice);
        cudaMemcpy(b0, nn.b[0].memptr(), size(b0_rows, b0_cols), cudaMemcpyHostToDevice);
        cudaMemcpy(W1, nn.W[1].memptr(), size(W1_rows, W1_cols), cudaMemcpyHostToDevice);
        cudaMemcpy(b1, nn.b[1].memptr(), size(b1_rows, b1_cols), cudaMemcpyHostToDevice);
    }

    void updateHostFromDevice(NeuralNetwork& nn) {
        cudaMemcpy(nn.W[0].memptr(), W0, size(W0_rows, W0_cols), cudaMemcpyDeviceToHost);
        cudaMemcpy(nn.b[0].memptr(), b0, size(b0_rows, b0_cols), cudaMemcpyDeviceToHost);
        cudaMemcpy(nn.W[1].memptr(), W1, size(W1_rows, W1_cols), cudaMemcpyDeviceToHost);
        cudaMemcpy(nn.b[1].memptr(), b1, size(b1_rows, b1_cols), cudaMemcpyDeviceToHost);
    }
   

    void copyDevice(double* to, double* from, int n_rows, int n_cols) {
        cudaMemcpy(to, from, size(n_rows, n_cols), cudaMemcpyDeviceToDevice);
    }
};

void parallel_feedforward(NeuralNetwork& nn, arma::mat& X, gpu_cache& gcache);

void parallel_backprop(gpu_cache& gcache, const arma::mat& y, double reg, struct grads& bpgrads);

void parallel_gradient_descent(gpu_cache& gc, struct grads& bpgrads, double learning_rate);

double norms(NeuralNetwork& nn) {
    double norm_sum = 0;

    for(int i = 0; i < nn.num_layers; ++i)  {
        norm_sum += arma::accu(arma::square(nn.W[i]));
    }

    return norm_sum;
}

void write_cpudata_tofile(NeuralNetwork& nn, int iter) {
    std::stringstream s;
    s << "Outputs/CPUmats/SequentialW0-" << iter << ".mat";
    nn.W[0].save(s.str(), arma::raw_ascii);
    std::stringstream t;
    t << "Outputs/CPUmats/SequentialW1-" << iter << ".mat";
    nn.W[1].save(t.str(), arma::raw_ascii);
    std::stringstream u;
    u << "Outputs/CPUmats/Sequentialb0-" << iter << ".mat";
    nn.b[0].save(u.str(), arma::raw_ascii);
    std::stringstream v;
    v << "Outputs/CPUmats/Sequentialb1-" << iter << ".mat";
    nn.b[1].save(v.str(), arma::raw_ascii);
}

void write_diff_gpu_cpu(NeuralNetwork& nn, int iter,
                        std::ofstream& error_file) {
    arma::mat A, B, C, D;

    std::stringstream s;
    s << "Outputs/CPUmats/SequentialW0-" << iter << ".mat";
    A.load(s.str(), arma::raw_ascii);
    double max_errW0 = arma::norm(nn.W[0]-A, "inf")/arma::norm(A, "inf");
    double L2_errW0  = arma::norm(nn.W[0]-A,2)/arma::norm(A,2);

    std::stringstream t;
    t << "Outputs/CPUmats/SequentialW1-" << iter << ".mat";
    B.load(t.str(), arma::raw_ascii);
    double max_errW1 = arma::norm(nn.W[1]-B, "inf")/arma::norm(B, "inf");
    double L2_errW1  = arma::norm(nn.W[1]-B,2)/arma::norm(B,2);

    std::stringstream u;
    u << "Outputs/CPUmats/Sequentialb0-" << iter << ".mat";
    C.load(u.str(), arma::raw_ascii);
    double max_errb0 = arma::norm(nn.b[0]-C, "inf")/arma::norm(C, "inf");
    double L2_errb0  = arma::norm(nn.b[0]-C,2)/arma::norm(C,2);

    std::stringstream v;
    v << "Outputs/CPUmats/Sequentialb1-" << iter << ".mat";
    D.load(v.str(), arma::raw_ascii);
    double max_errb1 = arma::norm(nn.b[1]-D, "inf")/arma::norm(D, "inf");
    double L2_errb1  = arma::norm(nn.b[1]-D,2)/arma::norm(D,2);

    int ow = 15;

    if(iter == 0) {
        error_file << std::left<< std::setw(ow) << "Iteration" << std::left<< std::setw(
                       ow) << "Max Err W0" << std::left << std::setw(ow) << "Max Err W1"
                   << std::left<< std::setw(ow) << "Max Err b0" << std::left<< std::setw(
                       ow) << "Max Err b1" << std::left << std::setw(ow) << "L2 Err W0" << std::left
                   << std::setw(ow) << "L2 Err W1" << std::left<< std::setw(
                       ow) << "L2 Err b0" << std::left<< std::setw(ow) << "L2 Err b1" << "\n";
    }

    error_file << std::left << std::setw(ow) << iter << std::left << std::setw(
                   ow) << max_errW0 << std::left << std::setw(ow) << max_errW1 <<
               std::left << std::setw(ow) << max_errb0 << std::left << std::setw(
                   ow) << max_errb1 << std::left<< std::setw(ow) << L2_errW0 << std::left <<
               std::setw(ow) << L2_errW1 << std::left << std::setw(ow) << L2_errb0 <<
               std::left<< std::setw(ow) << L2_errb1 << "\n";

}

/* CPU IMPLEMENTATIONS */
void feedforward(NeuralNetwork& nn, const arma::mat& X, struct cache& cache) {
    cache.z.resize(2);
    cache.a.resize(2);

    // std::cout << W[0].n_rows << "\n";tw
    assert(X.n_rows == nn.W[0].n_cols);
    cache.X = X;
    int N = X.n_cols;

    arma::mat z1 = nn.W[0] * X + arma::repmat(nn.b[0], 1, N);
    cache.z[0] = z1;

    arma::mat a1;
    sigmoid(z1, a1);
    cache.a[0] = a1;

    assert(a1.n_rows == nn.W[1].n_cols);
    arma::mat z2 = nn.W[1] * a1 + arma::repmat(nn.b[1], 1, N);
    cache.z[1] = z2;

    arma::mat a2;
    softmax(z2, a2);
    cache.a[1] = cache.yc = a2;
}

/*
 * Computes the gradients of the cost w.r.t each param.
 * MUST be called after feedforward since it uses the bpcache.
 * @params y : C x N one-hot column vectors
 * @params bpcache : Output of feedforward.
 * @params bpgrads: Returns the gradients for each param
 */
void backprop(NeuralNetwork& nn, const arma::mat& y, double reg,
              const struct cache& bpcache, struct grads& bpgrads) {
    bpgrads.dW.resize(2);
    bpgrads.db.resize(2);
    int N = y.n_cols;

    // std::cout << "backprop " << bpcache.yc << "\n";
    arma::mat diff = (1.0 / N) * (bpcache.yc - y);
    bpgrads.dW[1] = diff * bpcache.a[0].t() + reg * nn.W[1];
    bpgrads.db[1] = arma::sum(diff, 1);
    arma::mat da1 = nn.W[1].t() * diff;

    arma::mat dz1 = da1 % bpcache.a[0] % (1 - bpcache.a[0]);

    bpgrads.dW[0] = dz1 * bpcache.X.t() + reg * nn.W[0];
    bpgrads.db[0] = arma::sum(dz1, 1);
}

/*
 * Computes the Cross-Entropy loss function for the neural network.
 */
double loss(NeuralNetwork& nn, const arma::mat& yc, const arma::mat& y,
            double reg) {
    int N = yc.n_cols;
    double ce_sum = -arma::accu(arma::log(yc.elem(arma::find(y == 1))));

    double data_loss = ce_sum / N;
    double reg_loss = 0.5 * reg * norms(nn);
    double loss = data_loss + reg_loss;
    // std::cout << "Loss: " << loss << "\n";
    return loss;
}

/*
 * Returns a vector of labels for each row vector in the input
 */
void predict(NeuralNetwork& nn, const arma::mat& X, arma::rowvec& label) {
    struct cache fcache;
    feedforward(nn, X, fcache);
    label.set_size(X.n_cols);

    for(int i = 0; i < X.n_cols; ++i) {
        arma::uword row;
        fcache.yc.col(i).max(row);
        label(i) = row;
    }
}

/*
 * Computes the numerical gradient
 */
void numgrad(NeuralNetwork& nn, const arma::mat& X, const arma::mat& y,
             double reg, struct grads& numgrads) {
    double h = 0.00001;
    struct cache numcache;
    numgrads.dW.resize(nn.num_layers);
    numgrads.db.resize(nn.num_layers);

    for(int i = 0; i < nn.num_layers; ++i) {
        numgrads.dW[i].resize(nn.W[i].n_rows, nn.W[i].n_cols);

        for(int j = 0; j < nn.W[i].n_rows; ++j) {
            for(int k = 0; k < nn.W[i].n_cols; ++k) {
                double oldval = nn.W[i](j,k);
                nn.W[i](j, k) = oldval + h;
                feedforward(nn, X, numcache);
                double fxph = loss(nn, numcache.yc, y, reg);
                nn.W[i](j, k) = oldval - h;
                feedforward(nn, X, numcache);
                double fxnh = loss(nn, numcache.yc, y, reg);
                numgrads.dW[i](j, k) = (fxph - fxnh) / (2*h);
                nn.W[i](j, k) = oldval;
            }
        }
    }

    for(int i = 0; i < nn.num_layers; ++i) {
        numgrads.db[i].resize(nn.b[i].n_rows, nn.b[i].n_cols);

        for(int j = 0; j < nn.b[i].size(); ++j) {
            double oldval = nn.b[i](j);
            nn.b[i](j) = oldval + h;
            feedforward(nn, X, numcache);
            double fxph = loss(nn, numcache.yc, y, reg);
            nn.b[i](j) = oldval - h;
            feedforward(nn, X, numcache);
            double fxnh = loss(nn, numcache.yc, y, reg);
            numgrads.db[i](j) = (fxph - fxnh) / (2*h);
            nn.b[i](j) = oldval;
        }
    }
}

/*
 * Train the neural network &nn
 */
void train(NeuralNetwork& nn, const arma::mat& X, const arma::mat& y,
           double learning_rate, double reg,
           const int epochs, const int batch_size, bool grad_check, int print_every,
           int debug) {
    int N = X.n_cols;
    int iter = 0;
    int print_flag = 0;

    for(int epoch = 0 ; epoch < epochs; ++epoch) {
        int num_batches = (N + batch_size - 1)/batch_size;

        for(int batch = 0; batch < num_batches; ++batch) {
            int last_col = std::min((batch + 1)*batch_size-1, N-1);
            arma::mat X_batch = X.cols(batch * batch_size, last_col);
            arma::mat y_batch = y.cols(batch * batch_size, last_col);

            struct cache bpcache;
            feedforward(nn, X_batch, bpcache);

            struct grads bpgrads;
            backprop(nn, y_batch, reg, bpcache, bpgrads);

            if(print_every > 0 && iter % print_every == 0) {
                if(grad_check) {
                    struct grads numgrads;
                    numgrad(nn, X_batch, y_batch, reg, numgrads);
                    assert(gradcheck(numgrads, bpgrads));
                }

                std::cout << "Loss at iteration " << iter << " of epoch " << epoch << "/" <<
                          epochs << " = " << loss(nn, bpcache.yc, y_batch, reg) << "\n";
            }

            // Gradient descent step
            for(int i = 0; i < nn.W.size(); ++i) {
                nn.W[i] -= learning_rate * bpgrads.dW[i];
            }

            for(int i = 0; i < nn.b.size(); ++i) {
                nn.b[i] -= learning_rate * bpgrads.db[i];
            }

            /* Debug routine runs only when debug flag is set. If print_every is zero, it saves
               for the first batch of each epoch to avoid saving too many large files.
               Note that for the first time, you have to run debug and serial modes together.
               This will run the following function and write out files to CPUmats folder.
               In the later runs (with same parameters), you can use just the debug flag to
               output diff b/w CPU and GPU without running CPU version */
            if(print_every <= 0) {
                print_flag = batch == 0;
            } else {
                print_flag = iter % print_every == 0;
            }

            if(debug && print_flag) {
                write_cpudata_tofile(nn, iter);
            }

            iter++;
        }
    }
}

/*
 * TODO
 * Train the neural network &nn of rank 0 in parallel. Your MPI implementation
 * should mainly be in this function.
 */
void parallel_train(NeuralNetwork& nn, const arma::mat& X, const arma::mat& y,
                    double learning_rate, double reg,
                    const int epochs, const int batch_size, bool grad_check, int print_every,
                    int debug) {

    int rank, num_procs;
    MPI_SAFE_CALL(MPI_Comm_size(MPI_COMM_WORLD, &num_procs));
    MPI_SAFE_CALL(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    
    int M = (rank == 0) ? X.n_rows:0;
    int N = (rank == 0) ? X.n_cols:0;
    int num_class = (rank == 0) ? y.n_rows:0;
    MPI_SAFE_CALL(MPI_Bcast(&M, 1, MPI_INT, 0, MPI_COMM_WORLD));
    MPI_SAFE_CALL(MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD));
    MPI_SAFE_CALL(MPI_Bcast(&num_class, 1, MPI_INT, 0, MPI_COMM_WORLD));

    std::ofstream error_file;
    error_file.open("Outputs/CpuGpuDiff.txt");
    int print_flag = 0;

    /* HINT: You can obtain a raw pointer to the memory used by Armadillo Matrices
       for storing elements in a column major way. Or you can allocate your own array
       memory space and store the elements in a row major way. Remember to update the
       Armadillo matrices in NeuralNetwork &nn of rank 0 before returning from the function. */
    struct gpu_cache gc(nn, (batch_size + num_procs) / num_procs);
    gc.updateDeviceFromHost(nn);

    // pre divide batches
    int X_full, X_partial;
    int y_full, y_partial;
    int X_rows = M;
    int y_rows = num_class;
    int _batch_size, local_batch_size, last_local_batch_size;
    int num_batches = (N + batch_size - 1) / batch_size;
    int bs;
    arma::mat X_batch, y_batch;
    
    double *X_send;
    double *y_send;
    double *X_recv;
    double *y_recv;

    int *X_counts;
    int *y_counts;
    int *X_disp;
    int *y_disp;

    std::vector<arma::mat> X_batches(num_batches);
    std::vector<arma::mat> y_batches(num_batches);
    std::vector<int> batch_sizes(num_batches);
    std::vector<int> local_batch_sizes(num_batches);

    for (int batch = 0; batch < num_batches; ++batch) {
        if((batch + 1) * batch_size - 1 > N - 1) {
            _batch_size = N - batch * batch_size;
        } else {
            _batch_size = batch_size;
        }

        if (_batch_size % num_procs == 0) {
            local_batch_size = _batch_size / num_procs;
            last_local_batch_size = local_batch_size;
        } else {
            local_batch_size = (_batch_size + num_procs - 1) / num_procs;
            last_local_batch_size = _batch_size - local_batch_size * (num_procs - 1);
        }
            
        X_full = _batch_size * X_rows;
        y_full = _batch_size * y_rows;
        X_partial = local_batch_size * X_rows;
        y_partial = local_batch_size * y_rows;
        //double* X_send = new double[X_full];     //send buffer for full batch
        //double* y_send = new double[y_full];
        //double* X_recv = new double[X_partial];  //recv buffer for part of a batch
        //double* y_recv = new double[y_partial];
        X_send = new double[X_full];     //send buffer for full batch
        y_send = new double[y_full];
        X_recv = new double[X_partial];  //recv buffer for part of a batch
        y_recv = new double[y_partial];
  
        if (rank == 0) {
            int last_row = std::min((batch + 1) * batch_size - 1, N - 1);
            X_batch = X.cols(batch * batch_size, last_row);
            y_batch = y.cols(batch * batch_size, last_row);
            X_send = X_batch.memptr();
            y_send = y_batch.memptr();
        }

        //int *X_counts = new int[num_procs];
        //int *y_counts = new int[num_procs];
        //int *X_disp   = new int[num_procs];
        //int *y_disp   = new int[num_procs]; 
        X_counts = new int[num_procs];
        y_counts = new int[num_procs];
        X_disp   = new int[num_procs];
        y_disp   = new int[num_procs]; 

        for (int i = 0; i < num_procs; i++) {
            X_counts[i] = local_batch_size * X_rows;
            y_counts[i] = local_batch_size * y_rows;
            X_disp[i]   = i * local_batch_size * X_rows;
            y_disp[i]   = i * local_batch_size * y_rows;
        }
        // overwrite (last batch correction)
        X_counts[num_procs - 1] = last_local_batch_size * X_rows;
        y_counts[num_procs - 1] = last_local_batch_size * y_rows; 

        MPI_Scatterv(X_send, X_counts, X_disp, MPI_DOUBLE, 
                         X_recv, X_counts[rank],
                         MPI_DOUBLE, 0, MPI_COMM_WORLD); 

        MPI_Scatterv(y_send, y_counts, y_disp, MPI_DOUBLE, 
                         y_recv, y_counts[rank],
                         MPI_DOUBLE, 0, MPI_COMM_WORLD);

        if (rank != num_procs - 1) {
            bs = local_batch_size;
        } else {
            bs = last_local_batch_size;
        }
        arma::mat X_batch_local(X_recv, X_rows, bs, true, false);
        arma::mat y_batch_local(y_recv, y_rows, bs, true, false);
        X_batches[batch] = X_batch_local;
        y_batches[batch] = y_batch_local;
        local_batch_sizes[batch] = bs;
        batch_sizes[batch] = _batch_size;
    }

    /* iter is a variable used to manage debugging. It increments in the inner loop
       and therefore goes from 0 to epochs*num_batches */
    int iter = 0;
    for(int epoch = 0; epoch < epochs; ++epoch) {
        int num_batches = (N + batch_size - 1)/batch_size;
        for(int batch = 0; batch < num_batches; ++batch) {
            /*
             * Possible Implementation:
             * 1. subdivide input batch of images and `MPI_scatter()' to each MPI node
             * 2. compute each sub-batch of images' contribution to network coefficient updates
             * 3. reduce the coefficient updates and broadcast to all nodes with `MPI_Allreduce()'
             * 4. update local network coefficient at each node
             */
            
            // TODO call parallel functions to do feedforward and backprop
            gc.update_dims(local_batch_sizes[batch]);

            // Feedforward
            parallel_feedforward(nn, X_batches[batch], gc);

            // Backpropagation
            struct grads bpgrads;
            parallel_backprop(gc, y_batches[batch], reg, bpgrads);


            // AllReduce
            int count;
            for (int i = 0; i < nn.W.size(); i++) {
                count = nn.W[i].n_cols * nn.W[i].n_rows;
                bpgrads.dW[i] = bpgrads.dW[i] * ((1.0 * local_batch_sizes[batch]) / batch_sizes[batch]);
                MPI_Allreduce(MPI_IN_PLACE, bpgrads.dW[i].memptr(), count, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            }
            for(int i = 0; i < nn.b.size(); i++) {
                count = nn.b[i].n_cols * nn.b[i].n_rows;
                bpgrads.db[i] = bpgrads.db[i] * ((1.0 * local_batch_sizes[batch]) / batch_sizes[batch]);
                MPI_Allreduce(MPI_IN_PLACE, bpgrads.db[i].memptr(), count, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            }
           
            // Parallel gradient descent step
            parallel_gradient_descent(gc, bpgrads, learning_rate);

            if(print_every <= 0) {
                print_flag = batch == 0;
            } else {
                print_flag = iter % print_every == 0;
            }

            /* Following debug routine assumes that you have already updated the arma
               matrices in the NeuralNetwork nn.  */
            if(debug && rank == 0 && print_flag) {
                write_diff_gpu_cpu(nn, iter, error_file);
            }

            iter++;
        }
    }
    gc.updateHostFromDevice(nn);
    error_file.close();
}

//-------------- Functions needed for parallel implementation--------------------
/*
 * Feed forward with GPU matrix multiplications.
 */
void parallel_feedforward(NeuralNetwork& nn, arma::mat& X, gpu_cache& gc) {
    // update weights in gpu from the nn weights
    // gc.updateDeviceFromHost(nn);
    cudaMemcpy(gc.X, X.memptr(), gc.X_rows * gc.X_cols * sizeof(double), cudaMemcpyHostToDevice);

    // affine transformation 1 new:
    repColumn(gc.z0, gc.b0, gc.z0_rows, gc.z0_cols); 
    double alpha = 1.0;
    double beta  = 1.0;
    myGEMM4(gc.W0, gc.X, gc.z0,
           &alpha, &beta, gc.z0_rows, gc.z0_cols, gc.W0_cols); // matmul
    //(1000 x 784) (784 x 200) do gemm4

    // activation 1
    mySigmoid(gc.z0, gc.a0, gc.z0_rows, gc.z0_cols);

    // affine transformation 2
    repColumn(gc.z1, gc.b1, gc.z1_rows, gc.z1_cols);
    myGEMM2(gc.W1, gc.a0, gc.z1,
           &alpha, &beta, gc.z1_rows, gc.z1_cols, gc.W1_cols);
    //2 (10 x 1000) (1000 x 200) do gemm2

    // softmax
    mySoftmax(gc.z1, gc.a1, gc.z1_rows, gc.z1_cols);
    gc.copyDevice(gc.yc, gc.a1, gc.yc_rows, gc.yc_cols);
}

/*
 * Backpropagation with GPU matrix multiplications.
 */
void parallel_backprop(gpu_cache& gc, const arma::mat& y, double reg, struct grads& bpgrads) {
    bpgrads.dW.resize(2);
    bpgrads.db.resize(2);

    cudaMemcpy(gc.y, y.memptr(), gc.y_rows * gc.y_cols * sizeof(double), cudaMemcpyHostToDevice);
    elemwiseSubtract(gc.yc, gc.y, gc.yc_rows, gc.yc_cols);
    elemwiseMultiplyConstant(gc.yc, (1.0 / gc.y_cols), gc.yc_rows, gc.yc_cols);
    gc.copyDevice(gc.diff, gc.yc, gc.diff_rows, gc.diff_cols);

    cudaMemcpy(gc.dW1, gc.W1, gc.dW1_rows * gc.dW1_cols * sizeof(double), cudaMemcpyDeviceToDevice);
    transpose(gc.a0_t, gc.a0, gc.a0_rows, gc.a0_cols); 
    double alpha = 1.0;
    myGEMM3(gc.diff, gc.a0_t, gc.dW1, 
           &alpha, &reg, gc.dW1_rows, gc.dW1_cols, gc.diff_cols);
    //3 (10 x 200) (200 x 1000) do gemm 3

    arma::mat dW1(gc.dW1_rows, gc.dW1_cols);
    bpgrads.dW[1] = dW1;
    cudaMemcpy(bpgrads.dW[1].memptr(), gc.dW1, gc.dW1_rows * gc.dW1_cols * sizeof(double), cudaMemcpyDeviceToHost);

    arma::mat db1(gc.db1_rows, gc.db1_cols);
    sumRows(gc.db1, gc.diff, gc.diff_rows, gc.diff_cols);
    bpgrads.db[1] = db1;
    cudaMemcpy(bpgrads.db[1].memptr(), gc.db1, gc.db1_rows * gc.db1_cols * sizeof(double), cudaMemcpyDeviceToHost);

    transpose(gc.W1_t, gc.W1, gc.W1_rows, gc.W1_cols);
    double beta = 0.0;
    fill(gc.da0, 0, gc.da0_rows, gc.da0_cols); 
    myGEMM4(gc.W1_t, gc.diff, gc.da0,
           &alpha, &beta, gc.da0_rows, gc.da0_cols, gc.W1_t_cols);
    //4 (1000 x 10) (10 x 200) do gemm 4

    fill(gc.dz0, 1, gc.dz0_rows, gc.dz0_cols);
    elemwiseSubtract(gc.dz0, gc.a0, gc.dz0_rows, gc.dz0_cols);
    elemwiseMultiply(gc.dz0, gc.a0, gc.dz0_rows, gc.dz0_cols);
    elemwiseMultiply(gc.dz0, gc.da0, gc.dz0_rows, gc.dz0_cols);

    gc.copyDevice(gc.dW0, gc.W0, gc.dW0_rows, gc.dW0_cols);
    transpose(gc.X_t, gc.X, gc.X_rows, gc.X_cols);
    myGEMM4(gc.dz0, gc.X_t, gc.dW0, 
           &alpha, &reg, gc.dW0_rows, gc.dW0_cols, gc.dz0_cols);
    //5 (1000 x 200) (200 x 784) do gemm 4

    arma::mat dW0(gc.dW0_rows, gc.dW0_cols);
    bpgrads.dW[0] = dW0;
    cudaMemcpy(bpgrads.dW[0].memptr(), gc.dW0, gc.dW0_rows * gc.dW0_cols * sizeof(double), cudaMemcpyDeviceToHost);

    arma::mat db0(gc.db0_rows, gc.db0_cols);
    sumRows(gc.db0, gc.dz0, gc.dz0_rows, gc.dz0_cols);
    bpgrads.db[0] = db0;
    cudaMemcpy(bpgrads.db[0].memptr(), gc.db0, gc.db0_rows * gc.db0_cols * sizeof(double), cudaMemcpyDeviceToHost);
}

/*
 * Parallel gradient descent step with GPU. 
 */
void parallel_gradient_descent(gpu_cache& gc, struct grads& bpgrads, double learning_rate){
    // weight matrices
    cudaMemcpy(gc.dW0, bpgrads.dW[0].memptr(), gc.dW0_rows * gc.dW0_cols * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(gc.dW1, bpgrads.dW[1].memptr(), gc.dW1_rows * gc.dW1_cols * sizeof(double), cudaMemcpyHostToDevice);
    myGradientDescent(gc.W0, gc.dW0, gc.dW0_rows, gc.dW0_cols, learning_rate);
    myGradientDescent(gc.W1, gc.dW1, gc.dW1_rows, gc.dW1_cols, learning_rate);

    // bias vectors
    cudaMemcpy(gc.db0, bpgrads.db[0].memptr(), gc.db0_rows * gc.db0_cols * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(gc.db1, bpgrads.db[1].memptr(), gc.db1_rows * gc.db1_cols * sizeof(double), cudaMemcpyHostToDevice);
    myGradientDescent(gc.b0, gc.db0, gc.db0_rows, gc.db1_cols, learning_rate);
    myGradientDescent(gc.b1, gc.db1, gc.db1_rows, gc.db1_cols, learning_rate);
}
