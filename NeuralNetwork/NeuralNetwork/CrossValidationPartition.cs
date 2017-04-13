using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork
{

    public class RandomCrossValidationFactory
    {

        private int[] indeces;
        private int[] permutedInces;
        private Random rand;

        private readonly int numberOfPartitions;
        private readonly int totalNumberOfItems;
        private readonly int trainingSetSize;
        private readonly int testSetSize;

        public RandomCrossValidationFactory(int totalNumberOfItems, double trainingSetProportion, int numberOfPartitions)
        {
            // Initialize random number generator
            this.rand = new Random();

            // Initialize partition sizes
            this.numberOfPartitions = numberOfPartitions;
            this.totalNumberOfItems = totalNumberOfItems;
            initializeSetSizes(this.totalNumberOfItems, trainingSetProportion, 1.0 - trainingSetProportion, out this.trainingSetSize, out this.testSetSize);

            // Allocate indeces array
            this.indeces = new int[totalNumberOfItems];
            this.permutedInces = new int[totalNumberOfItems];
            for (int i = 0; i < totalNumberOfItems; i++)
            {
                this.indeces[i] = i;
                this.permutedInces[i] = i;
            }

            // Randomize indeces
            Permute();
        }

        public List<CrossValidationPartition> Create()
        {
            // Declare local variables
            int[] trainingSet;
            int[] testSet;

            List<CrossValidationPartition> partitions = new List<CrossValidationPartition>();
            for (int i = 0; i < numberOfPartitions; i++)
            {
                getSetIndeces(out trainingSet, out testSet);
                partitions.Add(new CrossValidationPartition(trainingSet, testSet));

                Permute();
            }

            return partitions;
        }

        public void Permute()
        {
            // Declare temporary variables
            int firstRandomIndex;
            int secondRandomIndex;
            int tempIndex;

            // Permute indeces
            for (int i = 0; i < indeces.Length; i++)
            {
                firstRandomIndex = rand.Next(0, permutedInces.Length);
                tempIndex = permutedInces[firstRandomIndex];
                secondRandomIndex = rand.Next(0, permutedInces.Length);

                // Swap indeces
                permutedInces[firstRandomIndex] = permutedInces[secondRandomIndex];
                permutedInces[secondRandomIndex] = tempIndex;
            }
        }

        private void initializeSetSizes(int numberOfItems, double trainingSetProportion, double testSetProportion,
                                        out int trainingSetSize, out int testSetSize)
        {
            // Round training and test sets
            trainingSetSize = (int)Math.Ceiling((double)numberOfItems * trainingSetProportion);
            testSetSize = numberOfItems - trainingSetSize;
        }

        private void getSetIndeces(out int[] trainingSet, out int[] testSet)
        {
            // Get set indeces
            trainingSet = getSubArray(permutedInces, 0, trainingSetSize);
            testSet = getSubArray(permutedInces, trainingSetSize, testSetSize);
        }

        // TODO [martin, 2017-04-06]: Refactor this using extension method and LINQ expression, e.g.: http://stackoverflow.com/questions/943635/getting-a-sub-array-from-an-existing-array
        private int[] getSubArray(int[] arr, int offset, int itemCount)
        {
            int[] subArray = new int[itemCount];
            for (int i = 0; i < itemCount; i++)
            {
                subArray[i] = arr[offset + i];
            }

            return subArray;
        }

    }

    public class CrossValidationPartition
    {
        private int[] trainingSet;
        private int[] testSet;

        public CrossValidationPartition(int[] trainingSet, int[] testSet)
        {
            this.trainingSet = trainingSet;
            this.testSet = testSet;
        }

        #region properties

        public int[] TrainingSet
        {
            get
            {
                return trainingSet;
            }
        }

        public int[] TestSet
        {
            get
            {
                return testSet;
            }
        }

        public int TrainingSetSize
        {
            get
            {
                return trainingSet.Length;
            }
        }

        public int TestSetSize
        {
            get
            {
                return testSet.Length;
            }
        }

        public int TotalNumberOfItems
        {
            get
            {
                return TrainingSetSize + TestSetSize;
            }
        }

        public double TrainingSetProportion
        {
            get
            {
                return (double)TrainingSetSize / (double)TotalNumberOfItems;
            }
        }

        public double TestSetProportion
        {
            get
            {
                return (double)TestSetSize / (double)TotalNumberOfItems;
            }
        }

        #endregion

        #region methods

        public void Write()
        {
            Console.Write("Training Set: ");
            for (int i = 0; i < TrainingSetSize; i++)
            {
                Console.Write("{0}, ", TrainingSet[i]);
            }
            Console.WriteLine();

            Console.Write("Test Set: ");
            for (int i = 0; i < TestSetSize; i++)
            {
                Console.Write("{0}, ", TestSet[i]);
            }
            Console.WriteLine();
        }

        #endregion

    }

}
