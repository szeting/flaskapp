{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\ASUS\\Anaconda3\\lib\\site-packages\\tensorflow\\python\\framework\\dtypes.py:526: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.\n",
      "  _np_qint8 = np.dtype([(\"qint8\", np.int8, 1)])\n",
      "C:\\Users\\ASUS\\Anaconda3\\lib\\site-packages\\tensorflow\\python\\framework\\dtypes.py:527: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.\n",
      "  _np_quint8 = np.dtype([(\"quint8\", np.uint8, 1)])\n",
      "C:\\Users\\ASUS\\Anaconda3\\lib\\site-packages\\tensorflow\\python\\framework\\dtypes.py:528: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.\n",
      "  _np_qint16 = np.dtype([(\"qint16\", np.int16, 1)])\n",
      "C:\\Users\\ASUS\\Anaconda3\\lib\\site-packages\\tensorflow\\python\\framework\\dtypes.py:529: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.\n",
      "  _np_quint16 = np.dtype([(\"quint16\", np.uint16, 1)])\n",
      "C:\\Users\\ASUS\\Anaconda3\\lib\\site-packages\\tensorflow\\python\\framework\\dtypes.py:530: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.\n",
      "  _np_qint32 = np.dtype([(\"qint32\", np.int32, 1)])\n",
      "C:\\Users\\ASUS\\Anaconda3\\lib\\site-packages\\tensorflow\\python\\framework\\dtypes.py:535: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.\n",
      "  np_resource = np.dtype([(\"resource\", np.ubyte, 1)])\n"
     ]
    }
   ],
   "source": [
    "import numpy as np\n",
    "import pandas as pd\n",
    "from sklearn.preprocessing import MinMaxScaler\n",
    "from werkzeug.wrappers import Request, Response\n",
    "from flask import Flask, request, jsonify, render_template\n",
    "from tensorflow.keras.models import load_model\n",
    "import pickle"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "WARNING:tensorflow:From C:\\Users\\ASUS\\Anaconda3\\lib\\site-packages\\tensorflow\\python\\ops\\resource_variable_ops.py:435: colocate_with (from tensorflow.python.framework.ops) is deprecated and will be removed in a future version.\n",
      "Instructions for updating:\n",
      "Colocations handled automatically by placer.\n",
      "WARNING:tensorflow:Sequential models without an `input_shape` passed to the first layer cannot reload their optimizer state. As a result, your model isstarting with a freshly initialized optimizer.\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      " * Running on http://localhost:9000/ (Press CTRL+C to quit)\n",
      "127.0.0.1 - - [12/Jun/2020 20:19:46] \"\u001b[37mGET / HTTP/1.1\u001b[0m\" 200 -\n",
      "127.0.0.1 - - [12/Jun/2020 20:19:46] \"\u001b[36mGET /static/style.css HTTP/1.1\u001b[0m\" 304 -\n",
      "127.0.0.1 - - [12/Jun/2020 20:19:47] \"\u001b[33mGET /favicon.ico HTTP/1.1\u001b[0m\" 404 -\n",
      "127.0.0.1 - - [12/Jun/2020 20:19:59] \"\u001b[37mPOST /predict HTTP/1.1\u001b[0m\" 200 -\n"
     ]
    }
   ],
   "source": [
    "app = Flask(__name__)\n",
    "\n",
    "model = load_model('model.pkl')\n",
    "scaler = pickle.load(open('scaler.pkl', 'rb'))\n",
    "cols=['Bedroom','Bathroom','SquareFeet','Carpark','Type','Title','Oth_Info','State']\n",
    "\n",
    "@app.route('/')\n",
    "def home():\n",
    "    return render_template('index.html')\n",
    "\n",
    "\n",
    "@app.route('/predict',methods=['POST'])\n",
    "def predict():\n",
    "        \n",
    "    int_features = [x for x in request.form.values()]\n",
    "    \n",
    "    final_features = np.array(int_features)\n",
    "    \n",
    "    data_unseen=pd.DataFrame([final_features],columns=cols)\n",
    "    \n",
    "    def type_label (row):\n",
    "        if row['Type'] == 'Apartments':\n",
    "            return 0\n",
    "        if row['Type'] == 'Houses' :\n",
    "            return 1\n",
    "\n",
    "    def title_label (row):\n",
    "        if row['Title'] == 'Freehold':\n",
    "            return 0\n",
    "        if row['Title'] == 'Leasehold' :\n",
    "            return 1\n",
    "\n",
    "    def oth1_label (row):\n",
    "        if row['Oth_Info'] == 'Non Bumi Lot':\n",
    "            return 0\n",
    "        if row['Oth_Info'] == 'Bumi Lot':\n",
    "            return 0\n",
    "        if row['Oth_Info'] == 'Malay Reserved' :\n",
    "            return 1\n",
    "\n",
    "    def oth2_label (row):\n",
    "        if row['Oth_Info'] == 'Malay Reserved':\n",
    "            return 0\n",
    "        if row['Oth_Info'] == 'Bumi Lot':\n",
    "            return 0\n",
    "        if row['Oth_Info'] == 'Non Bumi Lot' :\n",
    "            return 1\n",
    "\n",
    "    def state_label (row):\n",
    "        if row['State'] == 'Kuala Lumpur':\n",
    "            return 0\n",
    "        if row['State'] == 'Selangor' :\n",
    "            return 1\n",
    "\n",
    "    data_unseen['Type_ Houses']=data_unseen.apply (lambda row: type_label(row), axis=1)\n",
    "    data_unseen['Title_Leasehold']=data_unseen.apply (lambda row: title_label(row), axis=1)\n",
    "    data_unseen['Oth_Info_Malay Reserved']=data_unseen.apply (lambda row: oth1_label(row), axis=1)\n",
    "    data_unseen['Oth_Info_Non Bumi Lot']=data_unseen.apply (lambda row: oth2_label(row), axis=1)\n",
    "    data_unseen['State_Selongor']=data_unseen.apply (lambda row: state_label(row), axis=1)\n",
    "\n",
    "    data_unseen=data_unseen.drop(['Type','Title','Oth_Info','State'],axis=1)\n",
    "    \n",
    "    data_unseen=data_unseen.values\n",
    "    \n",
    "    #sample=np.array([4,3,2380,0,1,0,0,1,0])\n",
    "    \n",
    "    #data_unseen=scaler.fit_transform(data_unseen.reshape(-1,9))\n",
    "    data_unseen=scaler.transform(data_unseen.reshape(-1,9))\n",
    "    \n",
    "    prediction = model.predict(data_unseen)\n",
    "\n",
    "    output = int(prediction[0])\n",
    "\n",
    "    return render_template('index.html', prediction_text='The Predicted House Price is: RM {}'.format(output))\n",
    "\n",
    "\n",
    "if __name__ == '__main__':\n",
    "    from werkzeug.serving import run_simple\n",
    "    run_simple(\"localhost\",9000,app)\n",
    "   "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
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
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.1"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
