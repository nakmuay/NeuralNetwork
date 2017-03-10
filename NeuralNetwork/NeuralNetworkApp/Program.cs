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
            int[] layerSizes = { 3, 1 };

            TransferFunction[] transferFunctions =  {new SigmoidTransferFunction(),
                                                    new SigmoidTransferFunction()};

            BackPropagationNetwork bpn = new BackPropagationNetwork(2, layerSizes, transferFunctions);
            //bpn.Write();

            /*
            double input = { -1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0 };
            double[] output;
            double[] trainingOutput = { 1.0, 0.5625, 0.25, 0.0625, 0.0, 0.0625, 0.25, 0.5625, 1.0 };
            */

            double[] input = { 1.0, 0.5 };
            double[] output;
            double[] trainingOutput = { 0.25 };

            bpn.Run(ref input, out output);

            Console.ReadLine();
        }
    }
}
