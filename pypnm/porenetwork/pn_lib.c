#include <sys/types.h>
#include <assert.h>
#include <stdio.h>

int init_t_adj(const int *p_adj,int *pt_adj,int *edgelist, int nr_p,int *tube_count)
{
    int ind,ind_opp,p;
    int pi;
    int cnt;
    int p_opp=-1;

    *tube_count=0;
    for (ind =0; ind < nr_p; ++ind){
        for(p =0; p < 6; ++p){
            pi = p_adj[ind*6 +p];

            if(ind<pi){
                pt_adj[ind*6 +p] = *tube_count;
                edgelist[2*(*tube_count)]   = ind;
                edgelist[2*(*tube_count)+1] = pi;
                *tube_count= *tube_count + 1;
            }else if(pi>=0){
                ind_opp = pi;
                
                for (cnt=0; cnt<6; ++cnt){
                    if(p_adj[ind_opp*6+cnt]==ind) p_opp=cnt;
                }
                assert(p_opp>=0);
                pt_adj[ind*6+p] = pt_adj[ind_opp*6+p_opp];

            }
        }
    }

} 

