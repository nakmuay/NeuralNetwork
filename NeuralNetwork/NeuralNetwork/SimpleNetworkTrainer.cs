using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork
{
    public class SimpleNetworkTrainer
    {

        private static SimpleNetworkTrainer instance;

        private SimpleNetworkTrainer()
        {
        }

        public static SimpleNetworkTrainer Instance
        {
            get
            {
                if (instance == null)
                {
                    instance = new SimpleNetworkTrainer();
                }

                return instance;
            }
        }

        // TODO [martin, 2017-04-18]: Refactor training methods to share common code
        public TrainingInformation Train(BackPropagationNetwork net, IdentificationDataSet dataSet, TrainingOptions options)
        {
            // Prepare to train epoch
            TrainingInformation trainingInfo = new TrainingInformation();
            int iterations = 0;
            double errorSum = 0.0;
            do
            {
                // Train network
                iterations++;
                for (int i = 0; i < dataSet.Size; i++)
                {
                    var data = dataSet.Data[i];
                    for (int j = 0; j < data.NumSamples; j++)
                    {
                        errorSum += net.Train(data.InputData[j], data.OutputData[j], options.LearningRate, options.Momentum);
                    }
                }

                // Print some intermediate information
                if (iterations % 500 == 0)
                {
                    Console.WriteLine("Training epoch: {0}, error: {1:E2}", iterations, errorSum);
                }

                trainingInfo.IterationHistory.Add(iterations);
                trainingInfo.ErrorHistory.Add(errorSum);

            } while (errorSum > options.MaxError && iterations < options.MaxIterations);

            return trainingInfo;
        }

        public TrainingInformation Train(BackPropagationNetwork net, IdentificationData data, TrainingOptions options)
        {
            // Prepare to train epoch
            TrainingInformation trainingInfo = new TrainingInformation();
            int iterations = 0;
            double errorSum = 0.0;
            do
            {
                // Train network
                iterations++;
                for (int i = 0; i < data.NumSamples; i++)
                {
                    errorSum += net.Train(data.InputData[i], data.OutputData[i], options.LearningRate, options.Momentum);
                }

                // Print some intermediate information
                if (iterations % 500 == 0)
                {
                    Console.WriteLine("Training epoch: {0}, error: {1:E2}", iterations, errorSum);
                }

                trainingInfo.IterationHistory.Add(iterations);
                trainingInfo.ErrorHistory.Add(errorSum);

            } while (errorSum > options.MaxError && iterations < options.MaxIterations);

            return trainingInfo;
        }

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

    public class TrainingInformation
    {

        private List<int> iterationHistory;
        private List<double> errorHistory;

        public TrainingInformation()
        {
            iterationHistory = new List<int>();
            errorHistory = new List<double>();
        }

        #region properties

        public List<int> IterationHistory
        {
            get
            {
                return iterationHistory;
            }
            set
            {
                iterationHistory = value;
            }
        }

        public int NumIterations
        {
            get
            {
                return IterationHistory.Count;
            }
        }

        public int FinalIterationCount
        {
            get
            {
                return IterationHistory[IterationHistory.Count - 1];
            }
        }

        public List<double> ErrorHistory
        {
            get
            {
                return errorHistory;
            }
            set
            {
                errorHistory = value;
            }
        }

        public double FinalError
        {
            get
            {
                return ErrorHistory[ErrorHistory.Count - 1];
            }
        }

        #endregion

        #region methods

        public void WriteTrainingSummary()
        {
            Console.WriteLine("*** Training Summary (START) ***");
            Console.WriteLine("Final training error:               {0:E4}", FinalError);
            Console.WriteLine("Final number of traning iterations: {0}", FinalIterationCount);
            Console.WriteLine("*** Training Summary (END) ***");
        }

        public bool TrySerialize(string filename)
        {
            using (System.IO.StreamWriter file = new FormattingStreamWriter(@filename, System.Globalization.CultureInfo.InvariantCulture))
            {
                for (int i = 0; i < NumIterations; i++)
                {
                    file.WriteLine("{0}\t{1}", IterationHistory[i], ErrorHistory[i]);
                }
            }

            return false;
        }

        #endregion

    }

}