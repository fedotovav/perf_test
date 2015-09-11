BIN_DIR      = bin
OBJ_DIR      = obj
TEST_SRC_DIR = tests_src

TARGET = perf_test

CU_TEST_SRCS  = $(addprefix $(TEST_SRC_DIR)/, compute_cu.cu)
OMP_TEST_SRCS = $(addprefix $(TEST_SRC_DIR)/, compute_f.f08)
OCL_TEST_SRC  = $(addprefix $(TEST_SRC_DIR)/, compute_ocl.cpp)
CPP_SRCS = test.cpp main.cpp

OBJS  = $(addprefix $(OBJ_DIR)/, $(notdir $(OMP_TEST_SRCS:.f08=.o)))
OBJS += $(addprefix $(OBJ_DIR)/, $(notdir $(CU_TEST_SRCS:.cu=.o)))
OBJS += $(addprefix $(OBJ_DIR)/, $(notdir $(OCL_TEST_SRC:.cpp=.o)))
OBJS += $(addprefix $(OBJ_DIR)/, $(notdir $(CPP_SRCS:.cpp=.o)))

MODE_FLAGS = -g#-Ofast 

GF_FLAGS = -fopenmp -std=f2008 $(MODE_FLAGS)
CU_FLAGS = -gencode arch=compute_20,code=compute_20 -std=c++11
CPP_FLAGS = -std=c++11 $(MODE_FLAGS)
LINK_FLAGS = -fopenmp -lstdc++ -lgfortran -L/usr/local/cuda/lib64/ -lcuda -lcudart -lm -lOpenCL -L/usr/local/lib -lboost_program_options
CPP_INCLUDE = -I/usr/local/cuda/include -I/usr/local/include

all: dir $(TARGET)

dir: 
	if !(test -d $(BIN_DIR)); then mkdir $(BIN_DIR); fi
	if !(test -d $(OBJ_DIR)); then mkdir $(OBJ_DIR); fi

$(TARGET): $(OBJS)
	gcc -o $(BIN_DIR)/$(TARGET) $(OBJS) $(LINK_FLAGS)
 
$(OBJ_DIR)/%.o: %.cpp
	g++ -c $< -o $@ $(CPP_INCLUDE) $(CPP_FLAGS) 

$(OBJ_DIR)/%.o: $(TEST_SRC_DIR)/%.cpp
	g++ -c $< -o $@ $(CPP_INCLUDE) $(CPP_FLAGS) 

$(OBJ_DIR)/%.o: $(TEST_SRC_DIR)/%.cu
	/usr/local/cuda/bin/nvcc -c $< -o $@ $(CU_FLAGS)

$(OBJ_DIR)/%.o: $(TEST_SRC_DIR)/%.f08
	gfortran -c $< -o $@ $(GF_FLAGS)

clean:
	rm -rf $(BIN_DIR)/$(TARGET) $(OBJ_DIR)/*.o