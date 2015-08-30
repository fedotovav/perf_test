BIN_DIR = bin
OBJ_DIR = obj

TARGET = perf_test

CU_SRCS = compute_cu.cu
GF_SRCS = compute_f.f08
CPP_SRCS = compute_ocl.cpp main.cpp

OBJS  = $(addprefix $(OBJ_DIR)/, $(notdir $(GF_SRCS:.f08=.o)))
OBJS += $(addprefix $(OBJ_DIR)/, $(notdir $(CU_SRCS:.cu=.o)))
OBJS += $(addprefix $(OBJ_DIR)/, $(notdir $(CPP_SRCS:.cpp=.o)))

GF_FLAGS = -fopenmp -std=f2008 -Ofast 
CU_FLAGS = -gencode arch=compute_35,code=compute_35 -std=c++11
CPP_FLAGS = -std=c++11 -Ofast
LINK_FLAGS = -fopenmp -lstdc++ -lgfortran -L/usr/local/cuda/lib64/ -lcuda -lcudart -lm -lOpenCL
CPP_INCLUDE = -I/usr/local/cuda/include

all: dir $(TARGET)

dir: 
	if !(test -d $(BIN_DIR)); then mkdir $(BIN_DIR); fi
	if !(test -d $(OBJ_DIR)); then mkdir $(OBJ_DIR); fi

$(TARGET): $(OBJS)
	gcc -o $(BIN_DIR)/$(TARGET) $(OBJS) $(LINK_FLAGS)
 
$(OBJ_DIR)/%.o: %.cpp
	g++ -c $< -o $@ $(CPP_INCLUDE) $(CPP_FLAGS) 

$(OBJ_DIR)/%.o: %.cu
	/usr/local/cuda/bin/nvcc -c $< -o $@ $(CU_FLAGS)

$(OBJ_DIR)/%.o: %.f08
	gfortran -c $< -o $@ $(GF_FLAGS)

clean:
	rm -rf $(BIN_DIR)/$(TARGET) $(OBJ_DIR)/*.o