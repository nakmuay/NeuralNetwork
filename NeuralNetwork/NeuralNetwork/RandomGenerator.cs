using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork
{
    public class GaussianGenerator
    {

        private Random rng;
        private double mean;
        private double standardDeviation;

        public GaussianGenerator(double mean, double standardDeviation)
        {
            this.mean = mean;
            this.standardDeviation = standardDeviation;
            this.rng = new Random();
        }

        public GaussianGenerator(double mean, double standardDeviation, Random rand)
        {
            this.mean = mean;
            this.standardDeviation = standardDeviation;
            this.rng = rand;
        }

        public double NextDouble()
        {
            // We need to make sure that we do not generate a zero here since we are evaluating a log function later.
            var u1 = 1.0 - rng.NextDouble();
            var u2 = 1.0 - rng.NextDouble();

            double randStdNormal = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2);
            return mean + standardDeviation * randStdNormal;
        }

    }
}
