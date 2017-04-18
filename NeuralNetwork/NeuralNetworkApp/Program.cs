using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

using NeuralNetwork;
using System.IO;

namespace NeuralNetworkApp
{

    class Program
    {
        static void Main(string[] args)
        {
            // Create some training data
            IdenfificationDataFactory dataFactory = new TestIdentificationDataFactory(5, 100);
            IdentificationDataSet dataSet = dataFactory.GetData();

            // Create net
            int[] layerSizes = { 1, 8, 8, 1 };
            IDoubleEvaluatable[] activationFunction =  {new None(),
                                                        new TanhActivationFunction(),
                                                        new TanhActivationFunction(),
                                                        new Linear()};
            BackPropagationNetwork bpn = new BackPropagationNetwork(layerSizes, activationFunction, new SquaredErrorFunction());

            // Create network trainer
            SimpleNetworkTrainer trainer = SimpleNetworkTrainer.Instance;

            // Create trainging options
            TrainingOptions opt = new TrainingOptions();
            opt.MaxError = 1.0E-2;
            opt.MaxIterations = 20000;
            
            // Partition data
            RandomCrossValidationFactory cvFactory = new RandomCrossValidationFactory(dataSet.Size, 0.7, 10);
            List<CrossValidationPartition> cvPartition = cvFactory.Create();

            TrainingInformation trainInfo;
            double testError;
            foreach (var partition in cvPartition)
            {
                // Train network
                var trainingSet = dataSet.GetSubset(partition.TrainingSet);
                trainInfo = trainer.Train(bpn, trainingSet, opt);

                // Test network performance
                var testSet = dataSet.GetSubset(partition.TestSet);
                testError = bpn.Test(testSet);
            }


            IdentificationData testData = dataSet.Data[cvPartition[0].TestSet[0]];

            string outputFolder = Path.Combine(Directory.GetCurrentDirectory(), "NetworkData");
            Directory.CreateDirectory(outputFolder);
            string refFile = Path.Combine(outputFolder, "neural_net_reference.txt");
            testData.TrySerialize(refFile);

            // Write output before net has been trained
            IdentificationData beforeTrainData;
            bpn.Run(testData, out beforeTrainData);
            string beforeTrainFile = Path.Combine(outputFolder, "neural_net_before_training.txt");
            beforeTrainData.TrySerialize(beforeTrainFile);

            // Write output after net has been trained
            IdentificationData afterTrainData;
            bpn.Run(testData, out afterTrainData);
            string afterTrainFile = Path.Combine(outputFolder, "neural_net_after_training.txt");
            afterTrainData.TrySerialize(afterTrainFile);

            /*
            trainInfo.WriteTrainingSummary();

            // Write training information to file
            string trainingInfoFile = Path.Combine(outputFolder, "neural_net_training_info.txt");
            trainInfo.TrySerialize(trainingInfoFile);
            */

            Console.ReadLine();
        }
    }
}