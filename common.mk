#
# Copyright (C) 2011 by Justin Holewinski
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#

CUDA_TOOLKIT		:= /usr/local/cuda
CXX					  	:= g++
DEVICECXX	  		:= clang++

CXXFLAGS				:= -O3 -I$(CUDA_TOOLKIT)/include
LDFLAGS					:= -lcuda

CXXFLAGS_DEVICE	:= -O3 -S

# Command prefix for submitting batch jobs to a scheduler
SUBMIT					:=


# Detect architecture
ARCH						= $(shell uname -m)

ifeq ($(ARCH), x86_64)
	CXXFLAGS_DEVICE		+= -ccc-host-triple ptx64
	CXXFLAGS					+= -m64
else
	CXXFLAGS_DEVICE		+= -ccc-host-triple ptx32
	CXXFLAGS					+= -m32
endif


ifeq ($(verbose), 1)
	VERBOSE		  	:=
else
  VERBOSE     	:= @
endif



all	: $(NAME).x $(NAME).kernel.ptx

clean:
	@rm -f $(NAME).x $(NAME).kernel.ptx *.log

test	: all
	@echo "[TEST]"
	$(VERBOSE)$(SUBMIT) ./$(NAME).x

$(NAME).x	: $(NAME).cpp
	@echo "[HOST-C++]   $(NAME).cpp"
	$(VERBOSE)$(CXX) $(CXXFLAGS) $(NAME).cpp -o $(NAME).x $(LDFLAGS)

$(NAME).kernel.ptx	: $(NAME).kernel.cpp
	@echo "[DEVICE-C++] $(NAME).kernel.cpp"
	$(VERBOSE)$(DEVICECXX) $(CXXFLAGS_DEVICE) $(NAME).kernel.cpp -o $(NAME).kernel.ptx
