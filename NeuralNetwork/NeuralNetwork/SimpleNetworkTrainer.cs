using System;
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

        private int iterations;
        private double errorSum;
        private List<double> errorSumHistory;

        public SimpleNetworkTrainer(BackPropagationNetwork net, IdentificationData data)
        {
            this.net = net;
            this.data = data;

            this.iterations = 0;
            errorSumHistory = new List<double>();
        }


        public void Train(TrainingOptions options)
        {
            do
            {
                // Prepare to train epoch
                iterations++; errorSum = 0;
                for (int i = 0; i < data.NumSamples; i++)
                {
                    errorSum += net.Train(data.InputData[i], data.OutputData[i], options.LearningRate, options.Momentum);
                }

                // Print some intermediate information
                if (iterations % 100 == 0)
                {
                    Console.WriteLine("Training epoch: {0}, error: {1:E2}", iterations, errorSum);
                }

                /*
                if (iterations % 10000 == 0 && errorSum > options.MaxError)
                {
                    Console.WriteLine("Nudging synapse weights ...");
                    net.NudgeSynapses();
                }
                */

                errorSumHistory.Add(errorSum);

            } while (errorSum > options.MaxError && iterations < options.MaxIterations);
        }

        public void Train()
        {
            // Pass default training options
            TrainingOptions options = new TrainingOptions();
            Train(options);
        }


        #region properties

        public double ErrorSum
        {
            get
            {
                return this.errorSum;
            }
        }

        public double Iterations
        {
            get
            {
                return this.iterations;
            }
        }

        #endregion

    }

    public class TrainingOptions
    {

        private double learningRate;
        private double momentum;

        private double maxError;
        private int maxIterations;

        public TrainingOptions()
        {
            learningRate = 1.0E-2;
            momentum = 1.0E-3;
            maxError = 1.0E-3;
            maxIterations = 1000000;
        }


        #region properties

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

        public double Momentum
        {
            get
            {
                return momentum;
            }
            set
            {
                momentum = value;
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

        #endregion

    }

}
