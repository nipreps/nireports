<script>
var timestamp = Date.now()

function read_form() {
    var ds = "{{ metadata.dataset }}";
    var sub = "{{ metadata.filename }}";

    var artifacts = [];
    {% if config.components.artifacts %}
    $('#{{ config.components.artifacts.id }}-group input:checked').each(function() {
        artifacts.push($(this).attr('name'));
    });
    {% endif %}

    var rating = $('#{{ config.components.slider.id }}').val();
    var payload = {
        'dataset': ds,
        'subject': sub,
        'rating': rating,
        {% if config.components.artifacts %}
        'artifacts': artifacts,
        {% endif %}
        'time_sec': (Date.now() - timestamp) / 1000,
        'confidence': $('#{{ config.components.extra.id }}-confidence').val(),
        'comments': $('#{{ config.components.extra.id }}-comments').val()
    };

    var file = new Blob([JSON.stringify(payload)], {type: 'text/json'});
    $('#{{ config.components.actions[0].id }}').attr('href', URL.createObjectURL(file));
    $('#{{ config.components.actions[0].id }}').attr('download', payload['dataset'] + "_" + payload['subject'] + ".json");
    return payload
};

function toggle_rating() {
    if ($('#{{ config.id }}-menu').hasClass('d-none')) {
        $('#{{ config.id }}-menu').removeClass('d-none');
        $('#{{ config.id }}-toggler').prop('checked', true);
    } else {
        $('#{{ config.id }}-menu').addClass('d-none');
        $('#{{ config.id }}-toggler').prop('checked', false);
    }
};

$(window).on('load',function(){
    {% if config.components.actions[1] %}
    var authorization = $('#{{ config.components.actions[1].id }}').val()
    if (authorization.includes("secret_token")) {
        $('#{{ config.components.actions[1].id }}').addClass('d-none');
    };
    {% endif %}
    timestamp = Date.now();
});

</script>
<style type="text/css">
/* The slider itself */
.slider {
  -webkit-appearance: none;  /* Override default CSS styles */
  appearance: none;
  margin-bottom: 8px;
  margin-left: 10%;
  width: 80%;
  height: 5px; /* Specified height */
  background: #d3d3d3; /* Grey background */
  outline: none; /* Remove outline */
  opacity: 0.7; /* Set transparency (for mouse-over effects on hover) */
  -webkit-transition: .2s; /* 0.2 seconds transition on hover */
  transition: opacity .2s;
}

/* Mouse-over effects */
.slider:hover {
  opacity: 1; /* Fully shown on mouse-over */
}

.slider::-webkit-slider-thumb {
  -webkit-appearance: none;
  appearance: none;
  width: 25px;
  height: 25px;
  border: 0;
  background: url('https://raw.githubusercontent.com/nipreps/nireports/main/assets/slider-handle.png');
  cursor: pointer;
  z-index: 2000 !important;
}

.slider::-moz-range-thumb {
  width: 25px;
  height: 25px;
  border: 0;
  background: url('https://raw.githubusercontent.com/nipreps/nireports/main/assets/slider-handle.png');
  cursor: pointer;
  z-index: 2000 !important;
}

</style>
