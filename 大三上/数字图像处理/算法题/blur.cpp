#include <iostream>
#include <vector>
using namespace std;

vector< vector<int> > blur(vector< vector<int> >src,  int height_filter,int width_filter)
{
	vector<vector <int> > result;//定义二维数组
	vector<int> v;//定义一维数组
	float sum;
	int position_i,position_j;
	for (int i=0;i<src.size();i++)
	{
		v.clear();
		for (int j=0;j<src[i].size();j++)
		{
			sum=0;
			for (int m=-(height_filter+1)/2+1;m<(height_filter+1)/2;m++)
			{
				for (int n=-(width_filter+1)/2+1;n<(width_filter+1)/2;n++)
				{
					position_i = i+m;
					position_j = j+n;
					if(position_i<0 or position_j<0 or position_i>=src.size() or position_j >= src[i].size())
					{
						sum+=0;
					}
					else
					{
						sum+=src[position_i][position_j];
					}
				}
			}
			sum/=(height_filter*width_filter);
			sum = int(sum+0.5);
			v.push_back(sum);
		}
		result.push_back(v);
	}
	return result;
}

int main(int argc, char** argv) {
	int height_src,width_src,height_filter,width_filter;
	int temp=0;
	cin >> height_src>>width_src;
	cin >> height_filter>>width_filter;
	vector<vector <int> > src;//定义二维数组
	vector<vector <int> > result;//定义二维数组
	vector<int> v;//定义一维数组
	for (int i = 0; i < height_src; i++){//输入二维数组
		v.clear();//子数组返回时要清除
		for (int j = 0; j < width_src; j++){
			cin >> temp;
			v.push_back(temp);
		}
		src.push_back(v);
	}
//	for (int i = 0; i < src.size(); i++) {
//		for (int j = 0; j < src[i].size(); j++) {
//			cout << src[i][j]<<" " ;
//		}
//		cout << endl;
//	}
	result = blur(src,height_filter,width_filter);
	for (int i = 0; i < result.size(); i++) {
		for (int j = 0; j < result[i].size(); j++) {
			cout << result[i][j]<<" " ;
		}
		cout << endl;
	}
	return 0;
}


