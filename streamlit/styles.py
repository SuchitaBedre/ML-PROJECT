PAGE_STYLE = """

<style>


/* =========================
   APP BACKGROUND
========================= */


.stApp {

    background:
    linear-gradient(
        135deg,
        #fff7f0,
        #ffe4cc
    );

}



/* =========================
   CONTENT AREA
========================= */


.block-container {

    padding-top:2rem;
    padding-bottom:3rem;

}



/* =========================
   HEADINGS
========================= */


h1 {

    color:#ff6b35;
    font-weight:900;

}


h2 {

    color:#333;
    font-weight:800;

}


h3 {

    color:#555;
    font-weight:700;

}



/* =========================
   GLASS CARD
========================= */


.card {


    background:

    rgba(255,255,255,0.75);


    backdrop-filter:

    blur(10px);



    padding:30px;


    border-radius:25px;


    border:

    1px solid rgba(255,107,53,0.15);



    box-shadow:

    0px 10px 30px

    rgba(0,0,0,0.12);



    transition:0.3s;


}



.card:hover {


    transform:

    translateY(-8px);


    box-shadow:

    0px 15px 35px

    rgba(255,107,53,0.25);


}



/* =========================
   METRIC STYLE
========================= */


.metric-value {


font-size:38px;

font-weight:900;

color:#ff6b35;


}



.metric-title {


font-size:18px;

font-weight:600;

color:#666;


}



/* =========================
   RECIPE CARD
========================= */


.recipe-card {


background:white;


padding:25px;


border-radius:22px;


border-left:

6px solid #ff6b35;



box-shadow:

0px 8px 25px

rgba(0,0,0,0.12);



font-size:17px;


}



/* Recipe title */


.recipe-title {


color:#ff6b35;

font-size:28px;

font-weight:900;


}




/* =========================
   BUTTON
========================= */


.stButton button {


width:100%;


background:

linear-gradient(
90deg,
#ff6b35,
#ff914d
);



color:white;


border-radius:15px;


height:50px;


font-size:18px;


font-weight:800;


border:none;



box-shadow:

0px 5px 15px

rgba(255,107,53,0.35);


}



.stButton button:hover {


background:

linear-gradient(
90deg,
#e85d04,
#ff6b35
);



color:white;


transform:

scale(1.02);


}



/* =========================
   INPUT BOX
========================= */


.stTextInput input {


border-radius:15px;


border:

2px solid #ff914d;


font-size:17px;


}



/* =========================
   SIDEBAR
========================= */


section[data-testid="stSidebar"] {


background:


linear-gradient(
180deg,
#1f1f1f,
#111
);


}




section[data-testid="stSidebar"] h2 {


color:#ff6b35 !important;


}



section[data-testid="stSidebar"] * {


color:white;


}




/* =========================
   SUCCESS MESSAGE
========================= */


.stAlert {


border-radius:15px;


}



/* =========================
   DIVIDER
========================= */


hr {


border:

1px solid rgba(255,107,53,0.3);


}



</style>

"""