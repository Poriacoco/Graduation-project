{
 "cells": [
  {
   "cell_type": "code",
   "id": "e0a092f236a1392f",
   "metadata": {
    "ExecuteTime": {
     "end_time": "2025-05-16T04:24:21.900629Z",
     "start_time": "2025-05-16T04:24:21.894477Z"
    }
   },
   "source": [
    "import numpy as np\n",
    "import pandas as pd\n",
    "import torch\n",
    "from sklearn.preprocessing import MinMaxScaler\n",
    "import pandas as pd\n",
    "from sklearn.model_selection import KFold"
   ],
   "outputs": [],
   "execution_count": 2
  },
  {
   "metadata": {
    "ExecuteTime": {
     "end_time": "2025-05-16T04:41:23.192849Z",
     "start_time": "2025-05-16T04:41:23.186938Z"
    }
   },
   "cell_type": "code",
   "source": [
    "def load_data(file_path):\n",
    "    # 获取所有sheet的名称\n",
    "    sheet_names = pd.ExcelFile(file_path).sheet_names\n",
    "    # 读取所有sheet并存储在一个字典中\n",
    "    data_frames = {}\n",
    "    for sheet in sheet_names:\n",
    "        data_frames[sheet] = pd.read_excel(file_path, sheet_name=sheet)\n",
    "\n",
    "    for sheet, df in data_frames.items():\n",
    "        # 行列互换（转置）\n",
    "        df = df.transpose()\n",
    "\n",
    "        # 将转置后的DataFrame的第一行设为列名\n",
    "        df.columns = df.iloc[0]\n",
    "        df = df[1:]\n",
    "\n",
    "        df.reset_index(inplace=True, drop=False)\n",
    "        df.rename(columns={'index': '发酵周期/h'}, inplace=True)\n",
    "        df.columns.name = sheet\n",
    "\n",
    "        # 转换为数值类型，如果不能转换则设置为NaN\n",
    "        df = df.apply(pd.to_numeric, errors='coerce')\n",
    "\n",
    "        # 更新字典中的 DataFrame\n",
    "        if len(df.index) != 0:\n",
    "            data_frames[sheet] = df\n",
    "        else:\n",
    "            data_frames[sheet] = None\n",
    "\n",
    "    # 判断空值\n",
    "    df_all = []\n",
    "    for sheet, df in data_frames.items():\n",
    "        if len(df.columns[df.isnull().any()].tolist()) == 0:\n",
    "            df_all.append(df)\n",
    "\n",
    "    full_data = []\n",
    "    for data in df_all:\n",
    "        # 为指定列生成滞后特征\n",
    "        for column in ['酸钠', '残糖g/dl']:\n",
    "           data[f'{column}_diff'] = data[column].shift(-1)-data[column]\n",
    "        data=data.drop(data.index[-1])\n",
    "        full_data.append(data)\n",
    "\n",
    "    return full_data"
   ],
   "id": "78fda9d6981a52b1",
   "outputs": [],
   "execution_count": 30
  },
  {
   "metadata": {
    "collapsed": true,
    "ExecuteTime": {
     "end_time": "2025-05-16T04:31:44.770756Z",
     "start_time": "2025-05-16T04:31:44.766637Z"
    }
   },
   "cell_type": "code",
   "source": [
    "# 读取Excel文件\n",
    "def add_features(file_path = \"bio_all_34.xlsx\"):\n",
    "    remove_features = [\n",
    "        \"碱重kg\",  # 原始特征\n",
    "        \"重量KG\",  # 原始特征\n",
    "        \"PH值_差值\",  # 差值特征\n",
    "        \"罐压_差值\",  # 差值特征\n",
    "        \"风量L/h_差值\",  # 差值特征\n",
    "        \"转速r/min_差值\",  # 差值特征\n",
    "        \"温度_差值\"  # 差值特征\n",
    "    ]\n",
    "\n",
    "    sheets = pd.read_excel(file_path, sheet_name=None)\n",
    "    for sheet_name, df in sheets.items():\n",
    "        # 设置第一列为索引（特征名）\n",
    "        df = df.set_index(df.columns[0])\n",
    "        # 计算差值（沿列方向，即行内相邻单元格）\n",
    "        diff_df = df.diff(axis=1).iloc[:, 1:]\n",
    "\n",
    "        # 生成新行名并合并到原DataFrame\n",
    "        new_rows = []\n",
    "        for feature in df.index:\n",
    "            diff_row = pd.Series(\n",
    "                diff_df.loc[feature].values,\n",
    "                index=df.columns[1:],  # 差值从第二列开始\n",
    "                name=f\"{feature}_差值\"\n",
    "            )\n",
    "            new_rows.append(diff_row)\n",
    "\n",
    "        # 合并原数据和新行\n",
    "        updated_df = pd.concat([df, pd.DataFrame(new_rows)])\n",
    "\n",
    "        # 删除指定行（包括原始特征和差值特征）\n",
    "        updated_df = updated_df.drop(remove_features, errors=\"ignore\")\n",
    "        updated_df = updated_df.fillna(0)\n",
    "\n",
    "        # 重置索引并将特征名恢复为列\n",
    "        updated_df = updated_df.reset_index().rename(columns={\"index\": \"发酵周期\"})\n",
    "\n",
    "        # 更新当前工作表数据\n",
    "        sheets[sheet_name] = updated_df\n",
    "    # 保存到新Excel文件\n",
    "    with pd.ExcelWriter(\"bio_dif_tar.xlsx\") as writer:\n",
    "        for sheet_name, df in sheets.items():\n",
    "            df.to_excel(writer, sheet_name=sheet_name, index=False)\n",
    "\n"
   ],
   "id": "initial_id",
   "outputs": [],
   "execution_count": 21
  },
  {
   "metadata": {
    "ExecuteTime": {
     "end_time": "2025-05-16T04:32:01.351108Z",
     "start_time": "2025-05-16T04:32:01.021400Z"
    }
   },
   "cell_type": "code",
   "source": "full_data = add_features('bio_all_34.xlsx')",
   "id": "44d2ddd955a5722c",
   "outputs": [],
   "execution_count": 27
  },
  {
   "metadata": {
    "ExecuteTime": {
     "end_time": "2025-05-16T04:41:27.266495Z",
     "start_time": "2025-05-16T04:41:26.201674Z"
    }
   },
   "cell_type": "code",
   "source": "full_data = load_data('bio_dif_tar.xlsx')",
   "id": "5b11fccd53f754f7",
   "outputs": [],
   "execution_count": 31
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 2
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython2",
   "version": "2.7.6"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
