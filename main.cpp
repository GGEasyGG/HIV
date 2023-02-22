#include <fstream>
#include <cmath>
#include <algorithm>
#include <cstdlib>
#include <ctime>
#include <omp.h>
#include <vector>
#include <random>

#include "data.h"

using namespace std;

void Swap(double const &_t_cur, vector <unsigned long long> &_X, vector <double> &_Time, vector <double> &_result, unsigned int const &_m)
{
    for(unsigned int i=0; i<_m; i++)
    {
        _result[i] = _result[_m + i];
        _result[_m + i] = static_cast<double>(_X[i]);
    }

    _Time[0] = _Time[1];
    _Time[1] = _t_cur;
}

double interp2(double const &x1, double const &x2, double const &y1, double const &y2, double const &x)
{
    double pol, a, b, nx1, nx2, nx;

    nx1 = (x1 - x1) / (x2 - x1);
    nx2 = (x2 - x1) / (x2 - x1);
    nx = (x - x1) / (x2 - x1);

    a = (y2 - y1) / (nx2 - nx1);
    b = y1 - a * nx1;

    pol = a * nx + b;

    return pol;
}

int main()
{
    const unsigned int n = 84, m = 37, N = 10000, b = 20;
    const double pg_ml_i = 0.003489751, nM = 68, pg_ml_e = 3;
    double minimum, maximum;

    vector <double> final_result(56*m);
    vector <double> final_time(56);

    vector <double> intervals(56*N*m);
    vector <double> time_intervals(56);

    double t_max = 35;

    for(unsigned int i=0; i<56*m; i++)
    {
        final_result[i] = 0.0;
    }
    for(unsigned int i=0; i<56; i++)
    {
        final_time[i] = 0.0;
    }

    time_intervals[0] = 0.0;

    #pragma omp parallel
    {
        mt19937_64 mersenne(static_cast<unsigned int>(time(0)) + omp_get_thread_num());
        uniform_real_distribution<double> distribution(0.0,1.0);

        #pragma omp for
        for(unsigned int j=0; j<N; j++)
        {
            unsigned int k, g, l;
            double r1, r2, t, V_sum, f_Tat, f_Rev, TR, a, tmp, t_cur, t_last, tmp1, sum;

            vector <unsigned long long> X(m);
            vector <double> V(n);

            vector <double> result(2*m);
            vector <double> Time(2);

            l = 1;

            t_cur = 0.0;
            t_last = 0.0;

            g = 0;

            for(unsigned int i=0; i<m; i++)
            {
                X[i] = 0;
            }
            X[0] = 4;
            X[32] = 150;

            for(unsigned int i=0; i<m; i++)
            {
                result[g*m + i] = static_cast<double>(X[i]);
                #pragma omp atomic
                final_result[i] += result[i];
            }
            Time[g] = t_cur;
            #pragma omp atomic
            final_time[0] += Time[g];

            for(unsigned int i=0; i<m; i++)
            {
                intervals[j*56*m + i] = result[g*m + i];
            }

            g++;

            while(t_cur < t_max)
            {
                f_Tat = static_cast<double>(X[15])/(1000.0 + static_cast<double>(X[15]));
                f_Rev = static_cast<double>(X[16])/(77000.0 + static_cast<double>(X[16]));
                TR = 15.0 + 1500.0 * f_Tat;
                a = 1.0 - 0.9 * f_Rev;

                V_sum = 0.0;
                sum = 0.0;

                r1 = distribution(mersenne);
                r2 = distribution(mersenne);

                while(r1 == 0)
                {
                    r1 = distribution(mersenne);
                }

                while(r2 == 0)
                {
                    r2 = distribution(mersenne);
                }

                for(unsigned int i=0; i<n; i++)
                {
                    if(reacts[i] == 32)
                        V[i] = speeds[i] * static_cast<double>(X[reacts[i]])/pg_ml_e;
                    else if(reacts[i] == 31)
                        V[i] = speeds[i] * static_cast<double>(X[reacts[i]])/pg_ml_i;
                    else if((reacts[i] == 29) || (reacts[i] == 30) || (reacts[i] == 33))
                        V[i] = speeds[i] * static_cast<double>(X[reacts[i]])/nM;
                    else
                        V[i] = speeds[i] * static_cast<double>(X[reacts[i]]);
                }
                V[14] *= pg_ml_i; V[15] *= pg_ml_e; V[16] *= pg_ml_i;
                V[12] *= nM; V[13] *= nM; V[17] *= nM;
                V[53] *= nM; V[54] *= nM; V[55] *= pg_ml_i; V[56] *= pg_ml_i; V[60] *= nM;
                V[51] *= static_cast<double>(X[34]); V[52] *= static_cast<double>(X[36]);
                V[57] *= static_cast<double>(X[35]); V[58] *= static_cast<double>(X[34]);
                V[59] *= static_cast<double>(X[36]); V[64] *= static_cast<double>(X[35]);
                V[66] *= TR; V[67] *= a; V[68] *= a; V[69] *= f_Rev; V[70] *= f_Rev;
                for(unsigned int i=79; i<84; i++)
                {
                    V[i] *= static_cast<double>(X[17]);
                    V[i] *= static_cast<double>(X[18]);
                    V[i] *= static_cast<double>(X[19]);
                }

                for(unsigned int i=0; i<n; i++)
                {
                    V_sum += V[i];
                }

                t = -log(r1) * (1/V_sum);

                k = 0;

                for(unsigned int i=0; i<n; i++)
                {
                    sum += V[i];
                    if(sum > r2 * V_sum)
                    {
                        k = i;
                        break;
                    }
                }
                /*tmp = r2*V_sum - V[k];
                while((tmp > 0) && (k+1 < n))
                {
                    k++;
                    tmp -= V[k];
                }
                */
                t_cur += t;

                if(t_cur <= t_max)
                {
                    for(unsigned int i=0; i<m; i++)
                    {
                          X[i] += M[k][i];
                    }
                }
                else
                {
                    t_cur = t_max;
                }

                if(t_cur <= 5.0)
                {
                    if(g < 2)
                    {
                        for(unsigned int i=0; i<m; i++)
                        {
                            result[g*m + i] = static_cast<double>(X[i]);
                        }
                        Time[g] = t_cur;
                    }
                    else
                    {
                        if(!((Time[0] <= 0.2*l) && (Time[1] >= 0.2*l)))
                        {
                            Swap(t_cur, X, Time, result, m);
                        }
                    }

                    g++;
                }
                else if(t_cur == t_max)
                {
                    Swap(t_cur, X, Time, result, m);
                }
                else if((fabs(t_cur - t_last) >= 0.2) && ((k != 24) && (k != 25) && (k != 77) && (k != 78)))
                {
                    t_last = t_cur;

                    if(!((Time[0] <= static_cast<double>(l - b)) && (Time[1] >= static_cast<double>(l - b))))
                    {
                        Swap(t_cur, X, Time, result, m);
                    }
                }

                if(g >= 2)
                {
                    while((Time[0] <= 0.2*l) && (Time[1] >= 0.2*l) && (l <= 25))
                    {
                        for(unsigned int i=0; i<m; i++)
                        {
                           tmp1 = interp2(Time[0], Time[1], result[0*m + i], result[1*m + i], 0.2*l);
                           intervals[j * m * 56 + l * m + i] = tmp1;
                           #pragma omp atomic
                           final_result[l*m + i] += tmp1;
                        }
                        if(j == 0)
                            time_intervals[l] = (0.2 * l);
                        #pragma omp atomic
                        final_time[l] += (0.2 * l);

                        l++;
                    }

                    while((Time[0] <= static_cast<double>(l - b)) && (Time[1] >= static_cast<double>(l - b)) && ((l >= 26) && (l <= 54)))
                    {
                        for(unsigned int i=0; i<m; i++)
                        {
                           tmp1 = interp2(Time[0], Time[1], result[0*m + i], result[1*m + i], static_cast<double>(l - b));
                           intervals[j * m * 56 + l * m + i] = tmp1;
                           #pragma omp atomic
                           final_result[l*m + i] += tmp1;
                        }
                        if(j == 0)
                            time_intervals[l] = static_cast<double>(l - b);
                        #pragma omp atomic
                        final_time[l] += static_cast<double>(l - b);

                        l++;
                    }

                    if((l == 55) && (Time[1] == t_cur))
                    {
                        for(unsigned int i=0; i<m; i++)
                        {
                           tmp1 = result[1*m + i];
                           intervals[j * m * 56 + l * m + i] = tmp1;
                           #pragma omp atomic
                           final_result[l*m + i] += tmp1;
                        }
                        if(j == 0)
                            time_intervals[l] = static_cast<double>(l - b);
                        #pragma omp atomic
                        final_time[l] += static_cast<double>(l - b);

                        l++;
                    }
                }
            }

            printf("%d iteration success.\n", j+1);
        }
    }
    printf("All iterations success.\n");

    ofstream fout("matrix.txt");
    for(unsigned int i=0; i<=m-2; i++)
    {
        if(i != m-2)
            fout << i << " ";
        else
            fout << i;
    }
    fout << endl;

    for(unsigned int l=0; l<56; l++)
    {
        for(unsigned int i=0; i<m; i++)
        {
            if((i != 15) && (i != 16))
                fout << final_result[l*m + i]/N << " ";
        }
        fout << final_time[l]/N;
        fout << endl;
    }

    fout.close();

	ofstream fout6("interval_w_s.txt");

    for(unsigned int i=0; i<N; i++)
    {
        if(i != N-1)
            fout6 << i << " ";
        else
            fout6 << i;
    }
    fout6 << endl;

    for(unsigned int k=0; k<56; k++)
    {
        for(unsigned int i=0; i<m; i++)
        {
            if((i != 15) && (i != 16))
            {
                for(unsigned int j=0; j<N; j++)
                {
                    if(j != N-1)
                        fout6 << intervals[j * m * 56 + k * m + i] << " ";
                    else
                        fout6 << intervals[j * m * 56 + k * m + i];
                }
                fout6 << endl;
            }
        }
    }

    fout6.close();

    ofstream fout20("check.txt");

    for(unsigned int j=0; j<N; j++)
    {
        for(unsigned int k=0; k<56; k++)
        {
            if(((intervals[j * m * 56 + k * m + 31] > 0.0) || (intervals[j * m * 56 + k * m + 32] > 0.0)) && (k != 55))
            {
                for(unsigned int i=31; i<m; i++)
                {
                    fout20 << intervals[j * m * 56 + k * m + i] << " ";
                }
                fout20 << intervals[j * m * 56 + k * m + 27] << " " << time_intervals[k] << endl;
            }
        }

        for(unsigned int i=31; i<m; i++)
        {
            fout20 << intervals[j * m * 56 + 55 * m + i] << " ";
        }
        fout20 << intervals[j * m * 56 + 55 * m + 27] << " " << time_intervals[55] << endl;
        fout20 << endl;
    }

    fout20.close();

    double tmp;

    for(unsigned int i=0; i<m; i++)
    {
        for(unsigned int k=0; k<56; k++)
        {
            for(unsigned int j=0; j<N-1; j++)
            {
                for(unsigned int l=0; l<N-j-1; l++)
                {
                    if(intervals[l * m * 56 + k * m + i] > intervals[(l + 1) * m * 56 + k * m + i])
                    {
                        tmp = intervals[l * m * 56 + k * m + i];
                        intervals[l * m * 56 + k * m + i] = intervals[(l + 1) * m * 56 + k * m + i];
                        intervals[(l + 1) * m * 56 + k * m + i] = tmp;
                    }
                }
            }
        }
    }

    ofstream fout2("interval.txt");

    for(unsigned int i=0; i<N; i++)
    {
        if(i != N-1)
            fout2 << i << " ";
        else
            fout2 << i;
    }
    fout2 << endl;

    for(unsigned int k=0; k<56; k++)
    {
        for(unsigned int i=0; i<m; i++)
        {
            if((i != 15) && (i != 16))
            {
                for(unsigned int j=0; j<N; j++)
                {
                    if(j != N-1)
                        fout2 << intervals[j * m * 56 + k * m + i] << " ";
                    else
                        fout2 << intervals[j * m * 56 + k * m + i];
                }
                fout2 << endl;
            }
        }
    }

    fout2.close();

    ofstream fout3("intervals_min_max.txt");

    fout3 << 1 << " " << 2 << " " << 3 << endl;

    for(unsigned int i=0; i<m; i++)
    {
        if((i != 15) && (i != 16))
        {
            for(unsigned int k=0; k<56; k++)
            {
                maximum = intervals[k * m + i];
                minimum = intervals[k * m + i];

                for(unsigned int j=1; j<N; j++)
                {
                    if(intervals[j * m * 56 + k * m + i] >= maximum)
                        maximum = intervals[j * m * 56 + k * m + i];

                    if(intervals[j * m * 56 + k * m + i] <= minimum)
                        minimum = intervals[j * m * 56 + k * m + i];
                }

                fout3 << minimum << " " << maximum << " " << time_intervals[k] << endl;
            }
        }
    }

    fout3.close();

    ofstream fout1("time.txt");

    fout1 << "1" << endl;

    for(unsigned int i=0; i<35; i++)
    {
        for(unsigned int k=0; k<56; k++)
        {
            fout1 << time_intervals[k];
            fout1 << endl;
        }
    }

    fout1.close();

    return 0;
}
