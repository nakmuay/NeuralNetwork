﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork
{
    public class SimpleNetworkTrainer
    {

        private BackPropagationNetwork net;
        private IdentificationData data;
        private TrainingOptions options;

        private int iterations;
        private double error;
        private double[] errorHistory;

        public SimpleNetworkTrainer(BackPropagationNetwork net, IdentificationData data)
        {
            this.net = net;
            this.data = data;
            this.options = new TrainingOptions();

            this.iterations = 0;
            this.errorHistory = new double[data.NumSamples];
        }


        public void Train(TrainingOptions options)
        {
            do
            {
                // Prepare to train epoch
                iterations++; error = 0;
                for (int i = 0; i < data.NumSamples; i++)
                {
                    error += net.Train(data.InputData[i], data.OutputData[i], options.LearningRate);
                }

            } while (error > options.MaxError && iterations < options.MaxIterations);
        }

        public void Train()
        {
            Train(options);
        }


        public double Error
        {
            get
            {
                return this.error;
            }
        }

        public double Iterations
        {
            get
            {
                return this.iterations;
            }
        }

    }

    public class TrainingOptions
    {

        private double learningRate;

        private double maxError;
        private int maxIterations;

        public TrainingOptions()
        {
            learningRate = 0.01;
            maxError = 1.0E-3;
            maxIterations = 1000000;
        }


        public double LearningRate
        {
            get
            {
                return learningRate;
            }
            set
            {
                learningRate = value;
            }
        }

        public double MaxError
        {
            get
            {
                return maxError;
            }
            set
            {
                maxError = value;
            }
        }

        public int MaxIterations
        {
            get
            {
                return maxIterations;
            }
            set
            {
                maxIterations = value;
            }
        }

    }

}
