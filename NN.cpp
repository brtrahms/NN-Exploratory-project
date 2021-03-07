/*  @file   NN.cpp
 *
 *  @author Brandon Trahms
 *
 *  @brief  In this file, I implement a few simple neural networks that utilize forward and backward propagation.
 *          I run 3 tests which are embbeded in this file.
 *
 *          TEST #1
 *              This test is a simple game of Rock, Paper, Scissors. An input is given representing a Rock, Paper, or Scissors. 
 *              The neural network must then learn to give the winning response.
 *
 *          TEST #2
 *              In this test, a number from 1 to 9 are inputed and then the network must learn to identify which is even and odd.
 *
 *          TEST #3
 *              In this test, a binary number from 1 to 3 are inputed and then the network must learn to identify which is even and odd.
 *
 *          Each test runs til they reach 100% accuracy or 20000 iterations. Please be patient the tests may take a minute depending on the random initialized weights.
 */
#include <iostream>
#include <algorithm>
#include <vector>
#include <math.h>
#include <stdlib.h> 
#include <time.h>
#include <limits.h>
using namespace std;

double sigmoid(double in)
{
        return 1/(1 + exp(in*(-1)));
}

double dsigmoid(double in)
{
        return sigmoid(in)*(1 - sigmoid(in));
}
//Initializes the Neural Network
void NN(int L, int N[], double LR, vector<vector<double>> &Deltas,  vector<vector<vector<double>>> &Weights, vector<vector<double>> &Neurons, vector<vector<double>> &Cache)
{
        srand(time(NULL));

        for(int i = 0; i < L; i++)
        {
                vector<double> temp;
                
                for(int j = 0; j < N[i]; j++)
                {
                        temp.push_back(0);
                }
                Neurons.push_back(temp);
                Cache.push_back(temp);
                Deltas.push_back(temp);
        }

        for(int i = 0; i < L - 1; i++)
        {
                vector<vector<double>> temp;
                for(int j = 0; j < Neurons[i+1].size();j++)
                {
                        vector<double> t;
                        for(int k = 0; k < Neurons[i].size(); k++)
                        {
                                t.push_back(1);
                        }
                        temp.push_back(t);
                }
                Weights.push_back(temp);
        }
}
//Propagates Forward through the network given inputs
void propagateForward(vector<double> inputs, vector<vector<vector<double>>> &Weights, vector<vector<double>> &Neurons, vector<vector<double>> &Cache)
{
        for(int i = 0; i < Neurons.size(); i++)
        {
                for(int j = 0; j < Neurons[i].size(); j++)
                {
                        Neurons[i][j] = 0;
                }
        }

        for(int i = 0; i < Neurons[0].size(); i++)
                Neurons[0][i] = inputs[i];

        for(int i = 1; i < Neurons.size(); i++)
        {
                for(int k = 0; k < Weights[i - 1].size(); k++)
                {
                        for(int j = 0; j < Weights[i-1][k].size(); j++)
                        {
                                Neurons[i][k] += Neurons[i-1][j] * Weights[i-1][k][j];
                        }
                }
                
                for(int n = 0; n < Neurons[i].size(); n++)
                {
                        Cache[i][n] = Neurons[i][n];   
                        Neurons[i][n] = sigmoid(Neurons[i][n]);
                }
        }
}
//Propagates Backward through the network given a laerning rate and error in the delta vector
void propagateBackwards(double LR, vector<vector<double>> Deltas,  vector<vector<vector<double>>> &Weights, vector<vector<double>> &Neurons, vector<vector<double>> &Cache)
{
        for(int i = Neurons.size() - 2; i >= 0; i--)
        {
                for(int k = 0; k < Weights[i].size(); k++)
                {
                        for(int j = 0; j < Weights[i][k].size(); j++)
                        {
                                Deltas[i][j] +=  Deltas[i+1][k] * Weights[i][k][j];
                        }
                }
        }

        for(int i = 0; i < Neurons.size() - 1; i++)
        {
                for(int k = 0; k < Weights[i].size(); k++)
                {
                        for(int j = 0; j < Weights[i][k].size(); j++)
                        {
                                Weights[i][k][j] += LR * Deltas[i+1][k]*dsigmoid(Cache[i+1][k])*Neurons[i][j];
                        }
                }
        }
}
//tests accuracy of current test 1 nueral network
double test1(vector<vector<vector<double>>> Weights, vector<vector<double>> Neurons, vector<vector<double>> Cache)
{
        double batch = 1000, acc;
        srand(time(NULL));

        vector<double> in;
        vector<double> out;
        acc = 0;

        in.push_back(0);
        in.push_back(0);
        in.push_back(0);

        for(int i = 0; i < batch; i++)
        {
                in[0] = 0;
                in[1] = 0;
                in[2] = 0;

                int s = rand() % 3;
                in[s] = 1;
                propagateForward(in, Weights, Neurons, Cache);

                if(in[0] == 1)
                {
                        out.push_back(0);
                        out.push_back(1);
                        out.push_back(0);
                }else if(in[1] == 1)

                {
                        out.push_back(0);
                        out.push_back(0);
                        out.push_back(1);
                }else
                {
                        out.push_back(1);
                        out.push_back(0);
                        out.push_back(0);
                }

                int spot = 10;
                double max = INT_MIN;

                for(int d = 0; d < Neurons[Neurons.size()-1].size(); d++){
                        if(max < Neurons[Neurons.size()-1][d])
                        {
                                max = Neurons[Neurons.size()-1][d];
                                spot = d;
                        }else if(max == Neurons[Neurons.size()-1][d])
                        {
                                max = Neurons[Neurons.size()-1][d];
                                spot = 10;
                        }
                }

                if(s + 1 == spot || (s == 2 && spot == 0))
                        acc++;            
        }


        return acc/batch * 100;
}
//trains the test 1 nueral network with randomly generated inputs
void train1()
{
        vector<vector<vector<double>>> Weights;
        vector<vector<double>> Neurons;
        vector<vector<double>> Cache;
        vector<vector<double>> Deltas;
        double LR = .5;
        int batch = 1, bat = 0;
        srand(time(NULL));
        int L = 2;
        int N[L] = {3,3};
        bool complete = false;

        NN(L, N, LR, Deltas, Weights, Neurons, Cache);

        cout << "TEST #1: Rock, Paper, Scissors" << endl << "Network: ";

        for(int i = 0; i < L; i++)
                cout << N[i] << " ";

        cout << endl;
        cout << "Learning Rate: " << LR << endl;
        cout << "Batch: " << batch << endl;

        while(complete == false)
        {
                vector<double> in;
                vector<double> out;

                in.push_back(0);
                in.push_back(0);
                in.push_back(0);

                for(int i = 0; i < batch; i++)
                {
                        in[0] = 0;
                        in[1] = 0;
                        in[2] = 0;

                        int s = rand() % 3;
                        in[s] = 1;

                        propagateForward(in, Weights, Neurons, Cache);

                        if(in[0] == 1)
                        {
                                out.push_back(0);
                                out.push_back(1);
                                out.push_back(0);
                        }else if(in[1] == 1)
                        {
                                out.push_back(0);
                                out.push_back(0);
                                out.push_back(1);
                        }else
                        {
                                out.push_back(1);
                                out.push_back(0);
                                out.push_back(0);
                        }

                        double max = INT_MIN;
                        int spot = 10;

                        for(int d = 0; d < Deltas[L-1].size(); d++)
                                Deltas[L - 1][d] += out[d] - Neurons[L-1][d];

                }

                for(int d = 0; d < Deltas[L-1].size(); d++)
                        Deltas[L-1][d] = Deltas[L-1][d] / batch;

                propagateBackwards(LR, Deltas, Weights, Neurons, Cache);

                for(int d = 0; d < Deltas[L-1].size(); d++)
                        Deltas[L-1][d] = 0;

                bat++;

                int acc = test1(Weights, Neurons, Cache);

                if(acc == 100 || bat == 20000)
                {
                        complete = true;
                        cout << "After " << bat << " training iterations, test #1 has completed with " << acc << " percent accuarcy over 1000 tests.\n\n";
                }
        }
}
//tests the accuracy of the test 2 neural network
double test2(vector<vector<vector<double>>> Weights, vector<vector<double>> Neurons, vector<vector<double>> Cache)
{
        double batch = 1000, acc;
        srand(time(NULL));

        vector<double> in;
        vector<double> out;
        acc = 0;

        in.push_back(0);
        in.push_back(0);
        in.push_back(0);

        for(int i = 0; i < batch; i++)
        {
                for(int j = 0; j < in.size(); j++)
                        in[j] = 0;

                int s = rand() % in.size();
                in[s] = 1;

                propagateForward(in, Weights, Neurons, Cache);

                if((s + 1) % 2 == 0)
                {
                        out.push_back(1);
                        out.push_back(0);
                }
                else
                {
                        out.push_back(0);
                        out.push_back(1);
                }

                int spot = 10;
                double max = INT_MIN;

                for(int d = 0; d < Neurons[Neurons.size()-1].size(); d++){
                        if(max < Neurons[Neurons.size()-1][d])
                        {
                                max = Neurons[Neurons.size()-1][d];
                                spot = d;
                        }else if(max == Neurons[Neurons.size()-1][d])
                        {
                                max = Neurons[Neurons.size()-1][d];
                                spot = 10;
                        }
                }

                if(((s + 1) % 2 == 0 && spot == 0) || ((s+1) % 2 == 1 && spot == 1))
                        acc++;

        }


        return acc/batch * 100;
}
//trains the test 2 neural network with randomly generated inputs
void train2()
{
        vector<vector<vector<double>>> Weights;
        vector<vector<double>> Neurons;
        vector<vector<double>> Cache;
        vector<vector<double>> Deltas;
        double LR = .5;
        int batch = 1, bat = 0;
        srand(time(NULL));
        int L = 2;
        int N[L] = {9,2};
        bool complete = false;

        NN(L, N, LR, Deltas, Weights, Neurons, Cache);

        cout << "TEST #2: 1 to 9, Even or Odd?" << endl << "Network: ";

        for(int i = 0; i < L; i++)
                cout << N[i] << " ";

        cout << endl;
        cout << "Learning Rate: " << LR << endl;
        cout << "Batch: " << batch << endl;

        while(complete == false)
        {
                vector<double> in;
                vector<double> out;

                for(int j = 0; j < N[0]; j++)
                        in.push_back(0);

                for(int i = 0; i < batch; i++)
                {
                        for(int j = 0; j < N[0]; j++)
                                in[j] = 0;

                        int s = rand() % N[0];
                        in[s] = 1;

                        propagateForward(in, Weights, Neurons, Cache);

                        if((s + 1) % 2 == 0)
                        {
                                out.push_back(1);
                                out.push_back(0);
                        }
                        else
                        {
                                out.push_back(0);
                                out.push_back(1);
                        }

                        double max = INT_MIN;
                        int spot = 10;

                        for(int d = 0; d < Deltas[L-1].size(); d++)
                                Deltas[L - 1][d] += out[d] - Neurons[L-1][d];

                }

                for(int d = 0; d < Deltas[L-1].size(); d++)
                        Deltas[L-1][d] = Deltas[L-1][d] / batch;

                propagateBackwards(LR, Deltas, Weights, Neurons, Cache);

                for(int d = 0; d < Deltas[L-1].size(); d++)
                        Deltas[L-1][d] = 0;

                bat++;

                int acc = test2(Weights, Neurons, Cache);

                if(acc == 100 || bat == 20000)
                {
                        complete = true;
                        cout << "After " << bat << " training iterations, test #2 has completed with " << acc << " percent accuarcy over 1000 tests.\n\n";
                }
        }
}
//tests the accuracy of test 3 neural network
double test3(vector<vector<vector<double>>> Weights, vector<vector<double>> Neurons, vector<vector<double>> Cache)
{
        double batch = 1000, acc;
        srand(time(NULL));

        vector<double> in;
        vector<double> out;

        for(int j = 0; j < Neurons[0].size(); j++){
                in.push_back(0);
                out.push_back(0);
        }
        for(int i = 0; i < batch; i++)
        {
                for(int j = 0; j < Neurons[0].size(); j++)
                {
                        in[j] = 0;
                        out[j] = 0;
                }

                int s = rand() % 4 + 1;
                if(s == 1){
                        in[0] = 1;
                        out[1] = 1;
                }else if(s == 2){
                        in[1] = 1;
                        out[0] = 1;
                }else if(s == 3)
                {
                        in[0] = 1;
                        in[1] = 1;
                        out[1] = 1;
                }

                propagateForward(in, Weights, Neurons, Cache);

                int spot = 10;
                double max = INT_MIN;

                for(int d = 0; d < Neurons[Neurons.size()-1].size(); d++){
                        if(max < Neurons[Neurons.size()-1][d])
                        {
                                max = Neurons[Neurons.size()-1][d];
                                spot = d;
                        }else if(max == Neurons[Neurons.size()-1][d])
                        {
                                max = Neurons[Neurons.size()-1][d];
                                spot = 10;
                        }
                }

                if((s % 2 == 0 && spot == 0) || (s % 2 == 1 && spot == 1))
                        acc++;
        }

        return acc/batch * 100;
}
//trains the test 3 neural network with randomly generated data
void train3()
{
        vector<vector<vector<double>>> Weights;
        vector<vector<double>> Neurons;
        vector<vector<double>> Cache;
        vector<vector<double>> Deltas;
        double LR = .05;
        int batch = 1, bat = 0;
        srand(time(NULL));
        int L = 2;
        int N[L] = {2,2};
        bool complete = false;

        NN(L, N, LR, Deltas, Weights, Neurons, Cache);

        cout << "TEST #3: 1 to 3 in Binary, Even or Odd?" << endl << "Network: ";

        for(int i = 0; i < L; i++)
                cout << N[i] << " ";

        cout << endl;
        cout << "Learning Rate: " << LR << endl;
        cout << "Batch: " << batch << endl;

        while(complete == false)
        {
                vector<double> in;
                vector<double> out;

                for(int j = 0; j < N[0]; j++){
                        in.push_back(0);
                        out.push_back(0);
                }
                for(int i = 0; i < batch; i++)
                {
                        for(int j = 0; j < N[0]; j++)
                        {	
                                in[j] = 0;
                                out[j] = 0;
                        }

                        int s = rand() % 4 + 1;
                        if(s == 1){
                                in[0] = 1;
                                out[1] = 1;
                        }else if(s == 2){
                                in[1] = 1;
                                out[0] = 1;
                        }else if(s == 3)
                        {
                                in[0] = 1;
                                in[1] = 1;
                                out[1] = 1;
                        }

                        propagateForward(in, Weights, Neurons, Cache);

                        double max = INT_MIN;
                        int spot = 10;

                        for(int d = 0; d < Deltas[L-1].size(); d++)
                                Deltas[L - 1][d] += out[d] - Neurons[L-1][d];

                }

                for(int d = 0; d < Deltas[L-1].size(); d++)
                        Deltas[L-1][d] = Deltas[L-1][d] / batch;

                propagateBackwards(LR, Deltas, Weights, Neurons, Cache);

                for(int d = 0; d < Deltas[L-1].size(); d++)
                        Deltas[L-1][d] = 0;

                bat++;

                int acc = test3(Weights, Neurons, Cache);
        
                if(acc == 100 || bat == 20000)
                {
                        complete = true;
                        cout << "After " << bat << " training iterations, test #3 has completed with " << acc << " percent accuarcy over 1000 tests.\n\n";
                }
        }
}


int main()
{
        train1();
        train2();
        train3();
        return 0;
}
