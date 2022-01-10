#include <iostream>
#include <vector>
using namespace std;


vector< vector<int> > open_operator(vector< vector<int> >src,  int N,int M)
{
	vector<vector <int> > erosion;//定义二维数组
	vector<vector <int> > result;//定义二维数组
	vector<int> v;//定义一维数组
	float sum;
	int start_i=N/2,start_j=M/2;
	int flag=0,num;
	
	for (int i=0;i<src.size();i++)
	{
		v.clear();
		for (int j=0;j<src[i].size();j++)
		{
			num=0;
			
			if (src[i][j]==255 & i>=start_i & j>=start_j & i<src.size()-start_i & j < src[i].size()-start_j)
			{
				flag=0;
				for (int k=i-start_i;k<i+(N+1)/2;k++)
				{
					for (int h=j-start_j;h<j+(M+1)/2;h++)
					{
						if (src[k][h]!=255)
						{
							flag=1;
						}
					}
				}
				
				if(flag==0){
					num = 255;
				}
			}
			v.push_back(num);
		}
		erosion.push_back(v);
	}
	for (int i=0;i<src.size();i++)
	{
		for (int j=0;j<src[i].size();j++)
		{
			flag=0;
			num=0;
			for (int k=-N/2;k<=N/2;k++)
			{
				for (int h=-M/2;h<=M/2;h++)
				{
					if(i+k<0 | j+h<0|i+k>=src.size()|j+h>=src[i].size())
					{
						continue;
					}
					if(erosion[i+k][j+h] == 255)
					{
						flag=1;
						break;
					}
				}
			}
			if(flag==1)
			{
				num=255;
			}
			v.push_back(num);
			if(j == src[i].size()-1)
			{
				printf("%d\n",num);
			}
			else{
				printf("%d ",num);
			}
		}
	}
	
	return result;
}

int main(int argc, char** argv) {
	int N,M,H,W;
	int temp=0;
	cin >> N >> M;
	cin >> H >> W;
	vector<vector <int> > src;//定义二维数组
	vector<vector <int> > result;//定义二维数组
	vector<int> v;//定义一维数组
	for (int i = 0; i < H; i++){//输入二维数组
		v.clear();//子数组返回时要清除
		for (int j = 0; j < W; j++){
			cin >> temp;
			v.push_back(temp);
		}
		src.push_back(v);
	}
	open_operator(src,N,M);
	return 0;
}



