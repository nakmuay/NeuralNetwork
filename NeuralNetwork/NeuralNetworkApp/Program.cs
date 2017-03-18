using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

using NeuralNetwork;

namespace NeuralNetworkApp
{
    class Program
    {
        static void Main(string[] args)
        {
            int[] layerSizes = { 2, 3, 1 };

            ActivationFunction[] activationFunction =  {
                                                        new None(),
                                                        new SigmoidTransferFunction(),
                                                        new SigmoidTransferFunction()
                                                        };

            BackPropagationNetwork bpn = new BackPropagationNetwork(layerSizes, activationFunction);
            //bpn.Write();

            double[] input = { 1.0, 0.5 };
            double[] output;
            double[] wantedOutput = { 0.25 };

            //bpn.Run(ref input, out output);
            bpn.Train(input, wantedOutput);

            Console.ReadLine();
        }
    }
}
