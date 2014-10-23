language: PYTHON

variable {
 name: "learning_rate"
 size: 1
 type: FLOAT
 min:  0.001
 max:  0.01 
}

variable {
 name: "lr_exponential_decay_factor"
 size: 1
 type: FLOAT
 min: 1.0
 max: 1.1
}

#variable {
# name: "momentum_init"
# size: 1
# type: FLOAT
# min:  0.0
# max:  0.5
#}

#variable {
# name: "momentum_final"
# size: 1
# type: FLOAT
# min:  0.0
# max:  0.99
#}

variable {
 name: "h0_pattern_width"
 type: INT
 size: 1
 min:  1
 max:  45
}

variable {
 name: "h0_pool_size"
 type: INT
 size: 1
 min:  1
 max:  10
}

variable {
 name: "h0_patterns"
 type: INT
 size: 1
 min:  1
 max:  30
}
