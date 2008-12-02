#include <iostream>
#include "activationFunctions/sigmoidFunc.h"
#include "activationFunctions/heavisideStepFunc.h"
#include "activationFunctions/symmetricSigmoidFunc.h"

int main(int, char *[])
{
	NNLib::SigmoidFunc<float> sigm;
	std::cout << sigm(0) << std::endl;

	NNLib::HeavisideStepFunc<float> heav;
	std::cout << heav(0) << std::endl;

	NNLib::SymmetricSigmoidFunc<float> sym;
	std::cout << sym(0) << std::endl;

	return 0;
}