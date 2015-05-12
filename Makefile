###############################################################
## 				NeuralNet									###
##				        									###
##															###
##				Maurizio Idini								###
##															###
###############################################################

CXX		= g++ -std=c++11  

#sCXXFLAGS  	= -stdlib=libstdc++


TARGETS		= 	main


.PHONY: all clean cleanall
.SUFFIXES: .cpp 


%: %.cpp
	$(CXX) $(CXXFLAGS) -w -o neuralNet $<

all		: $(TARGETS)
clean		: 
	rm -f $(TARGETS)
cleanall	: clean
	\rm -f *.o *~
