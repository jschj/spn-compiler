//
// This file is part of the SPNC project.
// Copyright (c) 2020 Embedded Systems and Applications Group, TU Darmstadt. All rights reserved.
//
#include <iostream>

extern "C" {
  void trace(double d){
    std::cout << d << std::endl;
  }
}
