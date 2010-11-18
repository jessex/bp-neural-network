#include <stdio.h>
#include <stdlib.h>
#include "backprop.h"

int main(void) {
    init_rand();
    neural_network net; 
    neural_network *net_ptr = &net;
    initialize_network(net_ptr,2,2,1);
    
    train_network(net_ptr,4,10000,0.5,0.1);
    test_network(net_ptr,4);
    
    
    return 0;
}


