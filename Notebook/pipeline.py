"""Customer Feedback Theme Mining: end-to-end pipeline and HTML dashboard.

Run from the repository root with: python Notebook/pipeline.py
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "Reviews_3000.csv"
OUTPUT_DIR = ROOT / "Output"
RANDOM_STATE = 42
TOPICS = {
    1: "Food & Pet Products",
    2: "Chips & Snack Flavors",
    3: "Tea & Beverages",
    4: "Coffee Taste & Strength",
    5: "Pancake, Waffle & Baking Mixes",
}
STOPWORDS = set("""a about after all am an and any are as at be because been before being
between both but by can did do does doing down during each few for from further had has
have having he her here hers herself him himself his how i if in into is it its itself
just me more most my myself no nor not now of off on once only or other our ours ourselves
out over own same she so some such than that the their theirs them themselves then there
these they this those through to too under until up very was we were what when where which
while who whom why will with would you your yours yourself yourselves""".split())


def log(message: str) -> None:
    print(f"[pipeline] {message}")


def sentiment(score: int) -> str:
    return "Negative" if score <= 2 else "Neutral" if score == 3 else "Positive"


def clean_text(value: object) -> str:
    text = str(value or "").lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def fallback_lemma(word: str) -> str:
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 5 and word.endswith("ing"):
        return word[:-3]
    if len(word) > 4 and word.endswith("ed"):
        return word[:-2]
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def get_text_tools() -> tuple[set[str], object]:
    """Use NLTK when available without downloading or reinstalling packages."""
    try:
        from nltk.corpus import stopwords
        from nltk.stem import WordNetLemmatizer

        words = set(stopwords.words("english"))
        lemmatizer = WordNetLemmatizer()
        lemmatizer.lemmatize("reviews")
        return words, lemmatizer.lemmatize
    except (ImportError, LookupError):
        log("NLTK resources unavailable; using offline text-processing fallback.")
        return STOPWORDS, fallback_lemma


def preprocess(raw: pd.DataFrame) -> pd.DataFrame:
    stopwords, lemmatize = get_text_tools()
    df = raw.copy()
    df["Score"] = pd.to_numeric(df["Score"], errors="coerce").fillna(0).astype(int)
    df["Text"] = df["Text"].fillna("").astype(str)
    df["Summary"] = df["Summary"].fillna("").astype(str)
    df["Date"] = pd.to_datetime(df["Time"], unit="s", errors="coerce")
    df["Year"] = df["Date"].dt.year.fillna(0).astype(int)
    df["Month"] = df["Date"].dt.strftime("%Y-%m").fillna("Unknown")
    df["Sentiment"] = df["Score"].map(sentiment)
    df["Review_Length"] = df["Text"].str.len()
    df["Clean_Text"] = df["Text"].map(clean_text)
    df["Processed_Text"] = df["Clean_Text"].map(
        lambda text: " ".join(word for word in text.split() if word not in stopwords)
    )
    df["Final_Text"] = df["Processed_Text"].map(
        lambda text: " ".join(lemmatize(word) for word in text.split())
    )
    return df


def add_topics(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    vectorizer = TfidfVectorizer(max_features=5000, min_df=2, max_df=0.95)
    matrix = vectorizer.fit_transform(df["Final_Text"])
    nmf = NMF(n_components=5, random_state=RANDOM_STATE, init="nndsvda")
    weights = nmf.fit_transform(matrix)
    df = df.copy()
    df["Dominant_Topic"] = weights.argmax(axis=1) + 1
    df["Topic_Name"] = df["Dominant_Topic"].map(TOPICS)
    df["Topic_Confidence"] = weights.max(axis=1).round(6)

    features = vectorizer.get_feature_names_out()
    rows = []
    for topic, component in enumerate(nmf.components_, start=1):
        for rank, index in enumerate(component.argsort()[-10:][::-1], start=1):
            rows.append({"Topic": topic, "Topic_Name": TOPICS[topic], "Rank": rank,
                         "Keyword": features[index], "Weight": round(float(component[index]), 6)})
    keywords = pd.DataFrame(rows)
    evolution = pd.crosstab(df["Month"], df["Dominant_Topic"])
    for topic in TOPICS:
        if topic not in evolution:
            evolution[topic] = 0
    evolution = evolution[list(TOPICS)].rename(columns=lambda topic: f"Topic {topic}")
    return df, keywords, evolution.reset_index()


def score_models(df: pd.DataFrame) -> pd.DataFrame:
    train_x, test_x, train_y, test_y = train_test_split(
        df["Final_Text"], df["Sentiment"], test_size=0.2, random_state=RANDOM_STATE,
        stratify=df["Sentiment"]
    )
    vectorizer = TfidfVectorizer(max_features=5000, min_df=2, max_df=0.95)
    train_x = vectorizer.fit_transform(train_x)
    test_x = vectorizer.transform(test_x)
    models = {
        "Naive Bayes": MultinomialNB(),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "SVM": LinearSVC(random_state=RANDOM_STATE),
    }
    results = []
    for name, model in models.items():
        model.fit(train_x, train_y)
        prediction = model.predict(test_x)
        results.append({
            "Model": name,
            "Accuracy": round(float(accuracy_score(test_y, prediction)), 4),
            "Precision": round(float(precision_score(test_y, prediction, average="weighted", zero_division=0)), 4),
            "Recall": round(float(recall_score(test_y, prediction, average="weighted", zero_division=0)), 4),
            "F1-Score": round(float(f1_score(test_y, prediction, average="weighted", zero_division=0)), 4),
        })
    return pd.DataFrame(results)


def word_frequencies(df: pd.DataFrame) -> pd.DataFrame:
    counts = Counter(" ".join(df["Final_Text"]).split())
    frequencies = pd.DataFrame(counts.most_common(100), columns=["Word", "Frequency"])
    try:
        from wordcloud import WordCloud

        WordCloud(width=1200, height=600, background_color="white", max_words=100,
                  random_state=RANDOM_STATE).generate_from_frequencies(counts).to_file(
                      str(OUTPUT_DIR / "customer_feedback_wordcloud.png"))
    except ImportError:
        log("wordcloud is not installed; the HTML word cloud remains available.")
    return frequencies


def make_dashboard(data: dict[str, object]) -> str:
    encoded = json.dumps(data, ensure_ascii=True, separators=(",", ":"))
    return f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Customer Feedback Theme Mining</title><style>
:root{{--bg:#101825;--panel:#172235;--line:#304461;--text:#f4f7fb;--muted:#aab8cc;--cyan:#3dd6d0;--blue:#6096ff;--gold:#f3bc58;--pink:#ed719a;--green:#57c88b;--red:#ef6b73}}*{{box-sizing:border-box}}body{{margin:0;background:linear-gradient(145deg,#0c1420,#14263b);color:var(--text);font:14px Arial,sans-serif}}.shell{{max-width:1500px;margin:auto;padding:28px}}header{{display:flex;justify-content:space-between;gap:20px;align-items:end;margin-bottom:22px}}h1{{margin:0;font-size:28px}}.sub,.stamp,label,th{{color:var(--muted)}}.sub{{margin-top:7px}}.stamp{{text-align:right}}.filters,.kpis,.grid{{display:grid;gap:16px}}.filters{{grid-template-columns:repeat(3,1fr);margin-bottom:16px}}.filter,.panel,.kpi{{background:rgba(23,34,53,.92);border:1px solid var(--line);border-radius:8px;box-shadow:0 18px 40px rgba(0,0,0,.18)}}.filter,.panel,.kpi{{padding:16px}}label{{display:block;font-size:12px;margin-bottom:7px}}select{{width:100%;padding:9px;background:#101b2b;color:var(--text);border:0;border-radius:5px}}.kpis{{grid-template-columns:repeat(5,1fr);margin-bottom:16px}}.kpi{{min-height:100px}}.kpi span{{color:var(--muted);font-size:12px}}.kpi strong{{display:block;font-size:26px;margin-top:9px}}.kpi small{{color:var(--cyan);display:block;margin-top:5px}}.grid{{grid-template-columns:repeat(12,1fr)}}.wide{{grid-column:span 8}}.side{{grid-column:span 4}}.half{{grid-column:span 6}}.full{{grid-column:1/-1}}h2{{font-size:15px;margin:0 0 14px}}canvas{{display:block;width:100%;height:270px}}.cloud{{min-height:270px;display:flex;flex-wrap:wrap;align-content:center;justify-content:center;gap:7px 11px;overflow:hidden;padding:10px}}.cloud span{{line-height:1}}table{{width:100%;border-collapse:collapse}}th,td{{padding:10px 8px;border-bottom:1px solid var(--line);text-align:left}}th{{font-size:12px;font-weight:normal}}td.summary{{max-width:420px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}}.badge{{background:#263a57;border-radius:10px;padding:3px 7px;font-size:11px}}@media(max-width:1000px){{.kpis{{grid-template-columns:repeat(3,1fr)}}.wide,.side,.half{{grid-column:span 6}}}}@media(max-width:680px){{.shell{{padding:16px}}header{{display:block}}.stamp{{text-align:left;margin-top:10px}}.filters,.kpis{{grid-template-columns:1fr}}.wide,.side,.half{{grid-column:1/-1}}h1{{font-size:23px}}}}</style></head><body><main class="shell"><header><div><h1>Customer Feedback Theme Mining</h1><div class="sub">Interactive review intelligence dashboard</div></div><div class="stamp" id="stamp"></div></header><section class="filters"><div class="filter"><label>Sentiment</label><select id="sentiment"></select></div><div class="filter"><label>Theme</label><select id="topic"></select></div><div class="filter"><label>Score</label><select id="score"></select></div></section><section class="kpis"><article class="kpi"><span>Total reviews</span><strong id="total">-</strong><small>Current selection</small></article><article class="kpi"><span>Average score</span><strong id="average">-</strong><small>Out of 5.0</small></article><article class="kpi"><span>Positive sentiment</span><strong id="positive">-</strong><small>Of selected reviews</small></article><article class="kpi"><span>Leading theme</span><strong id="leading" style="font-size:18px">-</strong><small>Highest review volume</small></article><article class="kpi"><span>Best model</span><strong id="model" style="font-size:18px">-</strong><small id="modelnote">Highest F1 score</small></article></section><section class="grid"><article class="panel half"><h2>Score distribution</h2><canvas id="scores"></canvas></article><article class="panel half"><h2>Sentiment distribution</h2><canvas id="sentiments"></canvas></article><article class="panel wide"><h2>Theme distribution</h2><canvas id="topics"></canvas></article><article class="panel side"><h2>Frequent review terms</h2><div class="cloud" id="cloud"></div></article><article class="panel full"><h2>Theme evolution by month</h2><canvas id="evolution"></canvas></article><article class="panel half"><h2>Model performance</h2><canvas id="models"></canvas></article><article class="panel half"><h2>Recent matching reviews</h2><table><thead><tr><th>Score</th><th>Sentiment</th><th>Theme</th><th>Summary</th></tr></thead><tbody id="reviews"></tbody></table></article></section></main><script>
const D={encoded},P=['#3dd6d0','#6096ff','#f3bc58','#ed719a','#57c88b','#ef6b73'],I=id=>document.getElementById(id);function options(id,vs,label){{I(id).innerHTML='<option value="All">All '+label+'</option>'+vs.map(v=>'<option>'+v+'</option>').join('')}}function ctx(id){{let c=I(id),r=c.getBoundingClientRect(),d=devicePixelRatio||1;c.width=r.width*d;c.height=r.height*d;let x=c.getContext('2d');x.scale(d,d);return[x,r.width,r.height]}}function label(x,s,a,b,align='left',color='#aab8cc'){{x.fillStyle=color;x.textAlign=align;x.font='12px Arial';x.fillText(s,a,b)}}function bars(id,ls,vs){{let[x,w,h]=ctx(id),l=45,r=18,t=18,b=47,m=Math.max(...vs,1),cw=w-l-r,ch=h-t-b;x.clearRect(0,0,w,h);x.strokeStyle='#304461';for(let i=0;i<4;i++){{let y=t+ch*i/3;x.beginPath();x.moveTo(l,y);x.lineTo(w-r,y);x.stroke();label(x,Math.round(m*(3-i)/3),l-8,y+4,'right')}}let gap=cw/ls.length,bw=Math.max(8,gap*.62);ls.forEach((s,i)=>{{let bh=vs[i]/m*ch,px=l+i*gap+(gap-bw)/2;x.fillStyle=P[i%P.length];x.fillRect(px,t+ch-bh,bw,bh);x.save();x.translate(px+bw/2,t+ch+14);x.rotate(-.38);label(x,s,0,0,'right');x.restore()}})}}function donut(id,ls,vs){{let[x,w,h]=ctx(id),sum=vs.reduce((a,b)=>a+b,0)||1,cx=w*.38,cy=h*.5,r=Math.min(w,h)*.27,start=-Math.PI/2;x.clearRect(0,0,w,h);vs.forEach((v,i)=>{{let end=start+v/sum*Math.PI*2;x.beginPath();x.strokeStyle=P[i];x.lineWidth=28;x.arc(cx,cy,r,start,end);x.stroke();start=end}});x.fillStyle='#f4f7fb';x.font='bold 20px Arial';x.textAlign='center';x.fillText(sum.toLocaleString(),cx,cy+7);ls.forEach((s,i)=>{{let y=55+i*30;x.fillStyle=P[i];x.fillRect(w*.66,y-9,10,10);label(x,s+' '+Math.round(vs[i]/sum*100)+'%',w*.66+17,y)}})}}function lines(id,months,series){{let[x,w,h]=ctx(id),l=48,r=18,t=20,b=36,cw=w-l-r,ch=h-t-b,max=Math.max(...series.flatMap(s=>s.v),1);x.clearRect(0,0,w,h);x.strokeStyle='#304461';for(let i=0;i<4;i++){{let y=t+ch*i/3;x.beginPath();x.moveTo(l,y);x.lineTo(w-r,y);x.stroke();label(x,Math.round(max*(3-i)/3),l-7,y+4,'right')}}series.forEach((s,k)=>{{x.strokeStyle=P[k];x.lineWidth=2;x.beginPath();s.v.forEach((v,i)=>{{let px=l+(months.length===1?cw/2:cw*i/(months.length-1)),py=t+ch-v/max*ch;i?x.lineTo(px,py):x.moveTo(px,py)}});x.stroke();x.fillStyle=P[k];x.fillRect(l+k*76,7,9,9);label(x,s.n,l+13+k*76,16)}});months.forEach((m,i)=>{{if(i%Math.ceil(months.length/8)===0)label(x,m,l+cw*i/Math.max(1,months.length-1),h-10,'center')}})}}function groups(id,ms){{let[x,w,h]=ctx(id),l=44,r=18,t=20,b=44,cw=w-l-r,ch=h-t-b,keys=['Accuracy','Precision','Recall','F1-Score'];x.clearRect(0,0,w,h);x.strokeStyle='#304461';for(let i=0;i<5;i++){{let y=t+ch*i/4;x.beginPath();x.moveTo(l,y);x.lineTo(w-r,y);x.stroke();label(x,(1-i*.25).toFixed(2),l-7,y+4,'right')}}let g=cw/ms.length;ms.forEach((m,i)=>{{keys.forEach((k,j)=>{{let bw=g*.13,px=l+i*g+g*.12+j*bw*1.15;x.fillStyle=P[j];x.fillRect(px,t+ch-m[k]*ch,bw,m[k]*ch)}});label(x,m.Model,l+i*g+g/2,h-10,'center') }});keys.forEach((k,i)=>{{x.fillStyle=P[i];x.fillRect(l+i*105,7,9,9);label(x,k,l+13+i*105,16)}})}}function rows(){{return D.reviews.filter(r=>(I('sentiment').value==='All'||r.Sentiment===I('sentiment').value)&&(I('topic').value==='All'||r.Topic_Name===I('topic').value)&&(I('score').value==='All'||String(r.Score)===I('score').value))}}function render(){{let rs=rows(),n=rs.length,sc=[1,2,3,4,5],ss=['Positive','Neutral','Negative'],tc=D.topics.map(t=>rs.filter(r=>r.Topic_Name===t).length),sv=ss.map(s=>rs.filter(r=>r.Sentiment===s).length),best=[...D.modelPerformance].sort((a,b)=>b['F1-Score']-a['F1-Score'])[0],lead=D.topics[tc.indexOf(Math.max(...tc))]||'-';I('total').textContent=n.toLocaleString();I('average').textContent=n?(rs.reduce((a,r)=>a+r.Score,0)/n).toFixed(2):'-';I('positive').textContent=n?Math.round(sv[0]/n*100)+'%':'-';I('leading').textContent=n?lead:'-';I('model').textContent=best.Model;I('modelnote').textContent='F1 '+(best['F1-Score']*100).toFixed(1)+'%';bars('scores',sc.map(String),sc.map(s=>rs.filter(r=>r.Score===s).length));donut('sentiments',ss,sv);bars('topics',D.topics.map((_,i)=>'Theme '+(i+1)),tc);let ev=D.months.map(m=>D.topics.map(t=>rs.filter(r=>r.Month===m&&r.Topic_Name===t).length));lines('evolution',D.months,D.topics.map((t,i)=>({{n:'T'+(i+1),v:ev.map(a=>a[i])}})));groups('models',D.modelPerformance);let c=I('cloud');c.innerHTML='';D.words.slice(0,55).forEach((w,i)=>{{let s=document.createElement('span');s.textContent=w.Word;s.style.fontSize=(12+w.Frequency/D.words[0].Frequency*23)+'px';s.style.color=P[i%P.length];c.appendChild(s)}});I('reviews').innerHTML=rs.slice(0,7).map(r=>'<tr><td>'+r.Score+'</td><td><span class="badge">'+r.Sentiment+'</span></td><td>'+r.Topic_Name+'</td><td class="summary">'+(r.Summary||'No summary')+'</td></tr>').join('')||'<tr><td colspan="4">No reviews match the selected filters.</td></tr>'}}options('sentiment',['Positive','Neutral','Negative'],'sentiments');options('topic',D.topics,'themes');options('score',['1','2','3','4','5'],'scores');['sentiment','topic','score'].forEach(id=>I(id).onchange=render);I('stamp').textContent='Dataset: '+D.generatedAt+' | 3 ML models compared';render();window.onresize=render;
</script></body></html>'''


def export_results(df: pd.DataFrame, keywords: pd.DataFrame, evolution: pd.DataFrame,
                   performance: pd.DataFrame, frequencies: pd.DataFrame) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_columns = ["Id", "ProductId", "UserId", "ProfileName", "HelpfulnessNumerator",
                   "HelpfulnessDenominator", "Score", "Time", "Summary", "Text", "Date",
                   "Year", "Month", "Sentiment", "Review_Length", "Clean_Text",
                   "Processed_Text", "Final_Text", "Dominant_Topic", "Topic_Name", "Topic_Confidence"]
    df[all_columns].to_csv(OUTPUT_DIR / "output.csv", index=False)
    legacy = ["Id", "ProductId", "Score", "Summary", "Text", "Date", "Year", "Month",
              "Sentiment", "Review_Length", "Final_Text", "Dominant_Topic"]
    df[legacy].to_csv(OUTPUT_DIR / "processed_feedback.csv", index=False)
    keywords.to_csv(OUTPUT_DIR / "topic_keywords.csv", index=False)
    evolution.to_csv(OUTPUT_DIR / "topic_evolution.csv", index=False)
    performance.to_csv(OUTPUT_DIR / "model_performance.csv", index=False)
    frequencies.to_csv(OUTPUT_DIR / "word_frequencies.csv", index=False)
    data = {"generatedAt": pd.Timestamp.now().strftime("%d %b %Y %H:%M"),
            "topics": list(TOPICS.values()),
            "reviews": df[["Score", "Sentiment", "Topic_Name", "Month", "Summary"]].to_dict("records"),
            "months": evolution["Month"].tolist(), "modelPerformance": performance.to_dict("records"),
            "words": frequencies.head(80).to_dict("records")}
    (OUTPUT_DIR / "dashboard_data.json").write_text(json.dumps(data, ensure_ascii=True, indent=2), encoding="utf-8")
    (OUTPUT_DIR / "dashboard.html").write_text(make_dashboard(data), encoding="utf-8")


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")
    OUTPUT_DIR.mkdir(exist_ok=True)
    log(f"Loading {DATA_PATH.name}")
    df = preprocess(pd.read_csv(DATA_PATH))
    log("Building TF-IDF features and five NMF themes")
    df, keywords, evolution = add_topics(df)
    log("Training Naive Bayes, Logistic Regression, and SVM")
    performance = score_models(df)
    log("Writing CSV data and self-contained dashboard")
    export_results(df, keywords, evolution, performance, word_frequencies(df))
    log(f"Complete. Open {OUTPUT_DIR / 'dashboard.html'}")


if __name__ == "__main__":
    main()
