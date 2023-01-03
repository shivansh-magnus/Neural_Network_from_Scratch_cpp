#pragma once
#include "matrix.hpp"
#include <vector>
#include <cstdlib>
#include <cmath>

namespace sp
{
    inline float Sigmoid(float x)
    {
        return 1.0f / (1 + exp(-x));
    }

    inline float DSigmoid(float x)
    {
        return (x * (1 - x));
    }

    class SimpleNeuralNetwork
    {
        public:
            std::vector<uint32_t> _topology;
            std::vector<Matrix<float>> _weightMatrices;
            std::vector<Matrix<float>> _valueMatrices;
            std::vector<Matrix<float>> _biasMatrices;
            float _learningRate;
        public:
            SimpleNeuralNetwork(std::vector<uint32_t> topology,float learningRate = 0.1f)
                :_topology(topology),
                _weightMatrices({}),
                _valueMatrices({}),
                _biasMatrices({}),
                _learningRate(learningRate)
            {
                for(uint32_t i = 0; i < topology.size() - 1; i++)
                {
                    Matrix<float> weightMatrix(topology[i + 1], topology[i]); 
                    weightMatrix = weightMatrix.applyFunction([](const float &val){
                        return (float)rand() / RAND_MAX;
                    });
                    _weightMatrices.push_back(weightMatrix);
                    
                    Matrix<float> biasMatrix(topology[i + 1], 1);
                    biasMatrix = biasMatrix.applyFunction([](const float &val){
                        return (float)rand() / RAND_MAX;
                    });
                    _biasMatrices.push_back(biasMatrix);
   
                }
                _valueMatrices.resize(topology.size());
            }

            bool feedForword(std::vector<float> input)
            {
                if(input.size() != _topology[0])
                    return false;

                Matrix<float> values(input.size(), 1);
                for(uint32_t i = 0; i < input.size(); i++)
                    values._vals[i] = input[i];
                
                for(uint32_t i = 0; i < _weightMatrices.size(); i++)
                {  
                    _valueMatrices[i] = values;
                    values = values.multiply(_weightMatrices[i]);
                    values = values.add(_biasMatrices[i]);
                    values = values.applyFunction(Sigmoid);
                }
                _valueMatrices[_weightMatrices.size()] = values;
                return true;
            }

            bool backPropagate(std::vector<float> targetOutput)
            {
                if(targetOutput.size() != _topology.back())
                    return false;

                Matrix<float> errors(targetOutput.size(), 1);
                errors._vals = targetOutput;
                Matrix<float> sub=_valueMatrices.back().negetive();
                errors = errors.add(sub);
                
                for(int32_t i = _weightMatrices.size() - 1; i >= 0; i--)
                {
                    Matrix<float> trans = _weightMatrices[i].transpose();
                    Matrix<float> prevErrors = errors.multiply(trans);

                    Matrix<float> dOutputs = _valueMatrices[i + 1].applyFunction(DSigmoid);
                    Matrix<float> gradients = errors.multiplyElements(dOutputs);
                    gradients = gradients.multiplyScaler(_learningRate);
                    Matrix<float> weightGradients = _valueMatrices[i].transpose().multiply(gradients);
                    
                    _biasMatrices[i] = _biasMatrices[i].add(gradients);
                    _weightMatrices[i] = _weightMatrices[i].add(weightGradients);
                    errors = prevErrors;
                }
                
                return true;
            }
            
            std::vector<float> getPredictions()
            {
                return _valueMatrices.back()._vals;
            }

    };


}