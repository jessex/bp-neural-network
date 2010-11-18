typedef struct {
    int in_n, out_n, hid_n;
    double *in_a, *out_a, *hid_a;
    double **in_w, **out_w;
    double **in_c, **out_c;
} neural_network;


extern double patterns[4][2][2];


extern void init_rand();
extern double rand_base();
extern double rand_double(double min, double max);

extern double activate(double value);
extern double derivative(double value);
extern double **matrix(int x, int y, double **m);

extern void initialize_network(neural_network *net, int in, int hidden, int out);
extern void update_network(neural_network *net, double inputs[]);
extern void test_network(neural_network *net, int sets);

extern void train_network(neural_network *net, int sets, \
int iterations, double learn_rate, double momentum);

extern double back_propagate(neural_network *net, double learn_rate, \
double momentum, double goals[]);



