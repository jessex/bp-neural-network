#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "backprop.h"


double patterns[4][2][2] = { {{0,0},{0}} , {{0,1},{1}} , {{1,0},{1}} , {{1,1},{0}} };


/* ****************
 * RANDOM FUNCTIONS
 * **************** */ 

void init_rand() {
    srand((unsigned)(time(0)));
} 

double rand_base() {
    return rand()/((double)RAND_MAX + 1);
} 

double rand_double(double min, double max) {
    if (min>max) return rand_base()*(min-max)+max;
    else return rand_base()*(max-min)+min;
} 

/* *****************
 * UTILITY FUNCTIONS
 * ***************** */

double activate(double value) {
    return tanh(value);
}

double derivative(double value) {
    return 1.0 - (value * value);
}

double **matrix(int x, int y, double **m) {
    int row;
    m = malloc(x * sizeof(double *));
    for (row=0; row<x; row++) {
        m[row] = malloc(y * sizeof(double));
    }

    int i,j;
    for (i=0; i<x; i++) {
        for (j=0; j<y; j++) {
            m[i][j] = 0.0;
        }
    }
    return m;
}


/* *****************
 * NETWORK FUNCTIONS
 * ***************** */

void initialize_network(neural_network *net, int in, int hidden, int out) {
   
    //number of nodes per layer
    net->in_n = in + 1;
    net->out_n = out;
    net->hid_n = hidden;

    //node activations
    net->in_a = malloc((in+1) * sizeof(double));
    net->out_a = malloc(out * sizeof(double));
    net->hid_a = malloc(hidden * sizeof(double));
    int i;
    for (i=0; i<net->in_n; i++) net->in_a[i] = 0.0;
    for (i=0; i<net->out_n; i++) net->out_a[i] = 0.0;
    for (i=0; i<net->hid_n; i++) net->hid_a[i] = 0.0;
    
    //weight matrices for in and out layers
    net->in_w = matrix(net->in_n, net->hid_n, net->in_w);
    net->out_w = matrix(net->hid_n, net->out_n, net->out_w);
    //randomization of weight matrices
    int j;
    for (i=0; i<net->in_n; i++) 
        for (j=0; j<net->hid_n; j++) net->in_w[i][j] = rand_double(-0.2, 0.2);
    
    for (i=0; i<net->hid_n; i++) 
        for (j=0; j<net->out_n; j++) net->out_w[i][j] = rand_double(-2.0, 2.0);
    
    //most recent matrices
    net->in_c = matrix(net->in_n, net->hid_n, net->in_w);
    net->out_c = matrix(net->hid_n, net->out_n, net->out_w);
    
}

void update_network(neural_network *net, double inputs[]) {
    int i,j;
    //activate input nodes
    for (i=0; i<net->in_n; i++) net->in_a[i] = inputs[i];
    //activate hidden nodes
    for (i=0; i<net->hid_n; i++) {
        double sum = 0.0;
        for (j=0; j<net->in_n; j++) sum += net->in_a[j] * net->in_w[j][i];
        net->hid_a[i] = activate(sum);
    }
    //activate output nodes
    for (i=0; i<net->out_n; i++) {
        double sum = 0.0;
        for (j=0; j<net->hid_n; j++) sum += net->hid_a[j] * net->out_w[j][i];
        net->out_a[i] = activate(sum);
    }
}

void train_network(neural_network *net, int sets, \
int iterations, double learn_rate, double momentum) {
    int i,j,k;
    for (i=0; i<iterations; i++) {
        double error = 0.0;
        for (j=0; j<sets; j++) {
            double *in_set = patterns[j][0];
            double out[net->out_n];
            for (k=0; k<net->out_n; k++) out[k] = patterns[j][1][k];
            double *out_set = &out[0];
            update_network(net, in_set);
            error += back_propagate(net, learn_rate, momentum, out_set);
        }
        if ((i % 100) == 0) printf("error %-.5f\n", error);
    }
}


void test_network(neural_network *net, int sets) {
    int i,j,k;
    for (i=0; i<sets; i++) {
        double *in_set = patterns[i][0];
        int in = sizeof(in_set)/sizeof(double);
        for (j=0; j<in+1; j++) printf("%-.0f ", in_set[j]);
        printf("-> ");
        update_network(net, in_set);
        for (k=0; k<net->out_n; k++) printf("%-.8f ", net->out_a[k]);
        printf("\n");
    }
}

double back_propagate(neural_network *net, double learn_rate, \
double momentum, double goals[]) {
    int i,j;
    
    //output layer error terms (propagation)
    double *out_deltas = malloc(net->out_n * sizeof(double));
    for (i=0; i<net->out_n; i++)
        out_deltas[i] = derivative(net->out_a[i]) * (goals[i] - net->out_a[i]);
    
    //hidden layer error terms (propagation)
    double *hid_deltas = malloc(net->hid_n * sizeof(double));
    for (i=0; i<net->hid_n; i++) {
        double error = 0.0;
        for (j=0; j<net->out_n; j++) error += out_deltas[j] * net->out_w[i][j];
        hid_deltas[i] = derivative(net->hid_a[i]) * error;
    }
    
    //update output weights
    for (i=0; i<net->hid_n; i++) {
        for (j=0; j<net->out_n; j++) {
            double change = out_deltas[j] * net->hid_a[i];
            net->out_w[i][j] += (learn_rate * change) + \
            (momentum * net->out_c[i][j]);
            net->out_c[i][j] = change;
        }
    }
    
    //update input weights
    for (i=0; i<net->in_n; i++) {
        for (j=0; j<net->hid_n; j++) {
            double change = hid_deltas[j] * net->in_a[i];
            net->in_w[i][j] += (learn_rate * change) + \
            (momentum * net->in_c[i][j]);
            net->in_c[i][j] = change;
        }
    }
    
    free(out_deltas);
    free(hid_deltas);
    
    double error = 0.0;
    int goal_total = sizeof(goals)/sizeof(double);
    for (i=0; i<goal_total; i++) {
        double delta = goals[i] - net->out_a[i];
        error += 0.5 * (delta * delta);
    }
    return error;
}





