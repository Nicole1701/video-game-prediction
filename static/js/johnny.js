// Render & Populate Table Head
function render_thead(data){
    d3.select("table")
        .append("thead")
    var keys = Object.keys(data[0].vg_data[0])
    keys.forEach(key=>
        d3.select("thead")
            .append("th")
            // .append("p")
            // .attr("class","thtext")
            .text(key)
        )
}

// Render & Populate Table Body
function render_tbody(data){
    d3.select("table")
        .append("tbody")
    var keys = Object.keys(data[0].vg_data[0])
    var vg_data = data[0].vg_data
    console.log(vg_data.length)
    for(var i=0,vg_length=vg_data.length;i<vg_length;i++){
        var row_index = "row"+String(i)
        console.log(row_index)
        d3.select("tbody")
            .append("tr")
            .attr("id",row_index)
        for(var j=0,key_length=keys.length;j<key_length;j++){
            var row_id = "#"+String(row_index)
            var tdtext = vg_data[i][keys[j]]
            d3.select(row_id)
                .append("td")
                .text(tdtext)
        }
    }
}

// Run Function(s)
d3.json("vg_data").then(data=>
    render_thead(data)
    )
d3.json("vg_data").then(data=>
    render_tbody(data)
    )