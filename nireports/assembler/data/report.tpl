<?xml version="1.0" encoding="utf-8" ?>
<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Transitional//EN" "http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd">
<html xmlns="http://www.w3.org/1999/xhtml" xml:lang="en" lang="en">
<head>
<meta http-equiv="Content-Type" content="text/html; charset=utf-8" />
<meta name="generator" content="Docutils 0.12: http://docutils.sourceforge.net/" />
<title></title>
<script src="https://code.jquery.com/jquery-3.6.0.min.js" integrity="sha256-/xUj+3OJU5yExlq6GSYGSHk7tPXikynS7ogEvDej/m4=" crossorigin="anonymous"></script>

<style type="text/css">
.sub-report-title {}
.run-title {}

h1 { padding-top: 35px; }
h2 { padding-top: 20px; }
h3 { padding-top: 15px; }

.elem-desc {}
.elem-caption {
    margin-top: 15px
    margin-bottom: 0;
}
.elem-filename {}

div.elem-image {
  width: 100%;
  page-break-before:always;
}

.elem-image object.svg-reportlet {
    width: 100%;
    padding-bottom: 5px;
}
body {
    padding: 65px 10px 10px;
}

.boiler-html {
    font-family: "Bitstream Charter", "Georgia", Times;
    margin: 20px 25px;
    padding: 10px;
    background-color: #F8F9FA;
}

div#boilerplate pre {
    margin: 20px 25px;
    padding: 10px;
    background-color: #F8F9FA;
}

#errors div, #errors p {
    padding-left: 1em;
}
</style>

<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.2.3/dist/css/bootstrap.min.css" rel="stylesheet" integrity="sha384-rbsA2VBKQhggwzxH7pPCaAqO46MgnOM80zW1RWuH61DGLwZJEdK2Kadq2F9CUG65" crossorigin="anonymous">
<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.2.3/dist/js/bootstrap.bundle.min.js" integrity="sha384-kenU1KFdBIe4zVF0s0G1M5b4hcpxyD9F7jL+jjXkk+Q2h455rYXK/7HAuoJl+0I4" crossorigin="anonymous"></script>

</head>
<body>

<nav class="navbar fixed-top navbar-expand-lg bg-light">
<div class="container-fluid">
<div class="collapse navbar-collapse" id="navbarSupportedContent">
    <ul class="navbar-nav me-auto mb-2 mb-lg-0">
    {% for sub_report in sections %}
        {% if sub_report.isnested %}
        <li class="nav-item dropdown">
            <a class="nav-link dropdown-toggle" id="navbar{{ sub_report.name }}" role="button" data-bs-toggle="dropdown" aria-expanded="false" href="#{{sub_report.name}}">
            {{ sub_report.name }}
            </a>
            <ul class="dropdown-menu">
            {% for run_report in sub_report.reportlets %}
                {% if run_report.title %}
                <li><a class="dropdown-item" href="#{{run_report.name}}">{{run_report.title}}</a></li>
                {% endif %}
            {% endfor %}
            </ul>
        </li>
        {% else %}
        <li class="nav-item"><a class="nav-link" href="#{{sub_report.name}}">{{sub_report.name}}</a></li>
        {% endif %}
    {% endfor %}
    </ul>
</div>
</div>
</nav>
<noscript>
    <h1 class="text-danger"> The navigation menu uses Javascript. Without it this report might not work as expected </h1>
</noscript>

{% for sub_report in sections %}
    {% if sub_report.reportlets %}
    <div id="{{ sub_report.name }}">
    <h1 class="sub-report-title">{{ sub_report.name }}</h1>
    {% for run_report in sub_report.reportlets %}
        <div id="{{run_report.name}}">
            {% if run_report.title %}<h2 class="sub-report-group">{{ run_report.title }}</h2>{% endif %}
            {% if run_report.subtitle %}<h3 class="run-title">{{ run_report.subtitle }}</h3>{% endif %}
            {% if run_report.description %}<p class="elem-desc">{{ run_report.description }}</p>{% endif %}
            {% for elem in run_report.components %}
                {% if elem[0] %}
                    {% if elem[1] %}<p class="elem-caption">{{ elem[1] }}</p>{% endif %}
                    {{ elem[0] }}
                {% endif %}
            {% endfor %}
        </div>
    {% endfor %}
    </div>
    {% endif %}
{% endfor %}

<script type="text/javascript">
function toggle(id) {
    var element = document.getElementById(id);
    if(element.style.display == 'block')
        element.style.display = 'none';
    else
        element.style.display = 'block';
}

<!-- $(".nav .nav-link").on("click", function(){
   $(".nav").find(".active").removeClass("active");
   $(this).addClass("active");
}); -->
</script>
</body>
</html>
