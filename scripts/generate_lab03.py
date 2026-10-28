#!/usr/bin/env python3
"""Escribir el corpus de políticas y las preguntas del laboratorio 3 (respuestas con fuentes).

No consulta modelos: las siete políticas de Pacífico Sur, la aerolínea ficticia del
laboratorio 2, están escritas a mano y son sintéticas. La política de devoluciones tiene
una versión vigente y otra derogada, con plazos distintos, para practicar el filtro por
metadatos. Cada pregunta con respuesta trae los fragmentos literales que bastan para
responderla; el script comprueba que cada uno aparezca tal cual en su documento.
Tres preguntas no tienen respuesta en el corpus y miden la abstención.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "src/curso_ia/data/lab03"

# Cada documento: id, título, estado, vigencia y texto en Markdown (un párrafo por línea).
DOCUMENTS = [
    {
        "doc_id": "pol-equipaje",
        "title": "Política de equipaje",
        "status": "vigente",
        "valid_from": "2026-03-01",
        "valid_to": None,
        "text": """# Política de equipaje

Pacífico Sur · Vigente desde el 01/03/2026 · Aplica a vuelos nacionales e internacionales operados por Pacífico Sur.

## 1. Artículo personal y equipaje de mano

Todas las tarifas incluyen un artículo personal, como una cartera, una mochila pequeña o un bolso para notebook, de hasta 3 kg y que quepa bajo el asiento delantero (máximo 35 x 25 x 45 cm).

La tarifa Light incluye solo el artículo personal. Las tarifas Plus y Full incluyen además una maleta de mano de hasta 10 kg y de 55 x 35 x 25 cm como máximo, que se guarda en el compartimiento superior.

Quien viaje en tarifa Light y lleve una maleta de mano debe agregarla antes del vuelo en «Mis viajes» por CLP 12.000 por tramo. Si la maleta de mano se detecta en la puerta de embarque sin haber sido pagada, se cobra CLP 35.000 por tramo y se envía a la bodega.

En vuelos con alta ocupación, el personal puede pedir que una maleta de mano viaje en la bodega sin costo para el pasajero. En ese caso se entrega un comprobante y la maleta se retira en la cinta de equipaje del destino.

## 2. Equipaje facturado

La tarifa Full incluye una maleta facturada de hasta 23 kg. Las tarifas Light y Plus no incluyen equipaje facturado.

Una maleta facturada adicional cuesta CLP 22.000 por tramo si se compra en «Mis viajes» hasta 4 horas antes del vuelo, y CLP 40.000 por tramo si se compra en el aeropuerto.

El exceso de peso entre 23 y 32 kg tiene un cargo de CLP 30.000 por maleta y por tramo. No se aceptan maletas de más de 32 kg ni con una suma de largo, ancho y alto superior a 158 cm.

## 3. Equipaje especial

El equipo deportivo (bicicletas, tablas de surf, equipos de esquí y palos de golf) viaja como equipaje facturado y debe informarse al menos 24 horas antes del vuelo. Tiene un cargo fijo de CLP 45.000 por tramo, aunque la tarifa incluya una maleta facturada.

Los instrumentos musicales de hasta 10 kg pueden viajar en cabina si caben en el compartimiento superior; los de mayor tamaño requieren comprar un asiento adicional o viajar facturados en un estuche rígido.

## 4. Equipaje demorado, dañado o perdido

Si su maleta no llega en su vuelo o llega dañada, presente un reclamo en el mesón de equipaje del aeropuerto de destino antes de salir de la zona de llegadas. Se le entregará un número de reclamo (PIR) que identifica su caso.

Los daños deben reclamarse en el aeropuerto o, a más tardar, dentro de 7 días corridos desde la entrega de la maleta, adjuntando fotos del daño y la etiqueta de equipaje.

Mientras la maleta esté demorada fuera de su ciudad de residencia, Pacífico Sur reembolsa gastos de primera necesidad, como artículos de aseo y ropa básica, de hasta CLP 50.000 por día y por pasajero, con un máximo de 5 días y presentando las boletas.

Una maleta demorada se considera perdida si no se ubica dentro de 21 días corridos desde la fecha del vuelo. Desde ese momento puede solicitar la indemnización por pérdida, presentando una declaración del contenido y, si los tiene, los comprobantes de compra.

La indemnización por pérdida o daño se calcula según el valor declarado y comprobado, con los límites que fija la normativa aplicable a cada vuelo. Los objetos de valor, como joyas, dinero y equipos electrónicos, deben viajar en el equipaje de mano, porque su pérdida en el equipaje facturado no se indemniza salvo que se haya hecho una declaración de valor al facturar.
""",
    },
    {
        "doc_id": "pol-cambios",
        "title": "Política de cambios de vuelo",
        "status": "vigente",
        "valid_from": "2026-03-01",
        "valid_to": None,
        "text": """# Política de cambios de vuelo

Pacífico Sur · Vigente desde el 01/03/2026 · Aplica a cambios solicitados por el pasajero. Los cambios por cancelaciones o reprogramaciones de Pacífico Sur se rigen por la Política de retrasos y cancelaciones.

## 1. Familias tarifarias

Pacífico Sur vende tres familias tarifarias: Light, Plus y Full. Las condiciones de cada una se muestran al comprar y en el correo de confirmación de la reserva.

## 2. Cambios de fecha, hora o ruta

Los pasajes Light admiten un cambio de fecha u hora por tramo, con un cargo de CLP 25.000 más la diferencia tarifaria. No admiten cambios de ruta ni de nombre.

Los pasajes Plus admiten cambios de fecha, hora o ruta con un cargo de CLP 15.000 por tramo más la diferencia tarifaria.

Los pasajes Full admiten cambios de fecha, hora o ruta sin cargo; solo se paga la diferencia tarifaria, si la hay.

Todos los cambios deben hacerse hasta 3 horas antes de la salida del vuelo original en «Mis viajes» o en el centro de contacto. Después de ese plazo, el vuelo original se considera no utilizado.

La diferencia tarifaria es lo que cuesta, al momento del cambio, la nueva tarifa respecto de la pagada. Si la nueva tarifa es más barata, la diferencia no se devuelve en dinero.

## 3. Cambio el mismo día

Los pasajeros Plus y Full pueden adelantar o atrasar su vuelo dentro del mismo día y en la misma ruta, en el aeropuerto o en «Mis viajes», según disponibilidad. El cambio el mismo día cuesta CLP 10.000 en Plus y no tiene costo en Full, sin cobrar diferencia tarifaria. Los pasajes Light no admiten el cambio el mismo día.

## 4. Cambio de nombre y corrección de datos

No se permite transferir un pasaje a otra persona. Se puede corregir sin costo un error de hasta tres letras en el nombre o el apellido, siempre que la persona sea la misma que figura en su documento de identidad.

## 5. Mejora de cabina

Los vuelos nacionales con aviones A321 tienen una cabina Premium con asientos más amplios y embarque prioritario. La mejora a cabina Premium se ofrece en el aeropuerto según disponibilidad y su precio se informa en el mesón de atención; no es un cambio de tarifa y no modifica las condiciones de cambio del pasaje.

## 6. Pasajes comprados en agencias

Los pasajes comprados en una agencia de viajes se cambian en esa agencia, que puede cobrar sus propios cargos de servicio además de los de Pacífico Sur.

## 7. Pasajes con tramos de ida y vuelta

Cada tramo se cambia por separado y paga su propio cargo. Si cambia solo la ida, el pasaje de vuelta se mantiene válido, siempre que el vuelo de ida cambiado se utilice.

## 8. Saldo a favor

Cuando un pasaje no admite devolución en dinero, el valor no utilizado puede quedar como saldo a favor, si la tarifa lo permite. El saldo a favor es nominativo, se usa en «Mis viajes» para comprar otro pasaje de Pacífico Sur y vence 12 meses después de la fecha de emisión del pasaje original.
""",
    },
    {
        "doc_id": "pol-devoluciones-2026",
        "title": "Política de devoluciones",
        "status": "vigente",
        "valid_from": "2026-07-01",
        "valid_to": None,
        "text": """# Política de devoluciones

Pacífico Sur · Versión 3 · Vigente desde el 01/07/2026 · Reemplaza a la versión 2, vigente entre el 01/05/2024 y el 30/06/2026.

## 1. Alcance

Esta política regula la devolución del dinero de un pasaje cuando el pasajero decide no viajar. Las devoluciones por cancelaciones o retrasos de Pacífico Sur se rigen por la Política de retrasos y cancelaciones, pero se pagan según los plazos de la sección 5.

## 2. Qué se devuelve según la tarifa

Los pasajes Light no tienen devolución de la tarifa. Los pasajes Plus tienen devolución de la tarifa con un cargo de CLP 20.000 por tramo. Los pasajes Full tienen devolución completa de la tarifa, sin cargo.

Las tasas de embarque e impuestos se devuelven siempre, en todas las tarifas, aunque el pasaje no tenga devolución de la tarifa. Los servicios adicionales, como maletas o asientos, se devuelven solo si no se usaron y se compraron directamente en Pacífico Sur.

## 3. Derecho de retracto

Puede retractarse de una compra hecha en el sitio web o en la aplicación dentro de 24 horas desde la compra, con devolución completa y sin cargo, siempre que el vuelo salga al menos 7 días después de la compra. El retracto aplica a todas las tarifas, incluida Light.

## 4. Cómo y hasta cuándo solicitarla

La devolución se solicita en «Mis viajes», en la opción «Solicitar devolución», o en el centro de contacto. Las compras hechas en agencias de viajes se solicitan en la misma agencia.

Puede solicitar la devolución hasta un año después de la fecha de emisión del pasaje. Pasado ese plazo, solo se devuelven las tasas de embarque e impuestos.

Si el pasaje tiene tramos usados, se devuelve solo la parte no utilizada, calculada según la tarifa de cada tramo.

## 5. Plazos de pago

Una vez aprobada la solicitud, la devolución se paga en un plazo de 10 días hábiles. El pago se hace al mismo medio de pago usado en la compra; el tiempo en que el monto aparece en el estado de cuenta depende del banco emisor de la tarjeta.

Si la compra se pagó en efectivo o el medio de pago ya no está disponible, la devolución se transfiere a una cuenta bancaria a nombre del titular de la compra dentro del mismo plazo de 10 días hábiles.

Pacífico Sur responde cada solicitud en un plazo de 5 días hábiles, aprobándola o indicando qué antecedentes faltan.

## 6. Devolución en saldo a favor

En vez del dinero, el pasajero puede elegir recibir la devolución como saldo a favor, con un 10% adicional sobre el monto devuelto. El saldo a favor se rige por la Política de cambios de vuelo.

## 7. Casos especiales

En caso de fallecimiento del pasajero o de un familiar directo, o de una enfermedad grave que impida viajar y que se acredite con un certificado médico, los pasajes de cualquier tarifa tienen devolución completa sin cargo. La solicitud debe hacerse dentro del plazo de la sección 4 y acompañar el certificado correspondiente.
""",
    },
    {
        "doc_id": "pol-devoluciones-2024",
        "title": "Política de devoluciones (derogada)",
        "status": "derogada",
        "valid_from": "2024-05-01",
        "valid_to": "2026-06-30",
        "text": """# Política de devoluciones

Pacífico Sur · Versión 2 · Vigente entre el 01/05/2024 y el 30/06/2026 · Derogada: reemplazada por la versión 3 desde el 01/07/2026.

## 1. Alcance

Esta política regula la devolución del dinero de un pasaje cuando el pasajero decide no viajar o cuando su vuelo es cancelado o reprogramado por Pacífico Sur.

## 2. Qué se devuelve según la tarifa

Los pasajes Light no tienen devolución de la tarifa. Los pasajes Plus tienen devolución de la tarifa con un cargo de CLP 30.000 por tramo. Los pasajes Full tienen devolución de la tarifa con un cargo de CLP 10.000 por tramo.

Las tasas de embarque e impuestos se devuelven con un cargo administrativo de CLP 8.000 por pasaje. Los servicios adicionales, como maletas o asientos, no se devuelven.

## 3. Cómo y hasta cuándo solicitarla

La devolución se solicita en el centro de contacto o en las oficinas de venta, con el número de pasaje y el documento de identidad del pasajero.

Puede solicitar la devolución hasta seis meses después de la fecha de emisión del pasaje. Pasado ese plazo, el pasaje no tiene devolución.

Si el pasaje tiene tramos usados, se devuelve la diferencia entre lo pagado y el valor de una tarifa de un solo tramo para el tramo usado.

## 4. Plazos de pago

Las devoluciones aprobadas se pagan dentro de 30 días hábiles desde la solicitud. Las devoluciones a tarjetas de crédito pueden tardar hasta dos estados de cuenta en reflejarse.

Las compras pagadas en efectivo se devuelven con un vale de compra para un nuevo pasaje, válido por 6 meses.

## 5. Cancelaciones de Pacífico Sur

Si Pacífico Sur cancela su vuelo, puede pedir la devolución completa del pasaje, sin cargo, hasta 30 días después de la fecha del vuelo cancelado.

## 6. Casos especiales

En caso de fallecimiento del pasajero, se devuelve la tarifa sin cargo a sus herederos, presentando el certificado de defunción y la posesión efectiva. Las enfermedades no dan derecho a devolución.
""",
    },
    {
        "doc_id": "pol-retrasos",
        "title": "Política de retrasos y cancelaciones",
        "status": "vigente",
        "valid_from": "2026-03-01",
        "valid_to": None,
        "text": """# Política de retrasos y cancelaciones

Pacífico Sur · Vigente desde el 01/03/2026 · Aplica cuando Pacífico Sur retrasa, reprograma o cancela un vuelo.

## 1. Aviso a los pasajeros

Pacífico Sur informa los cambios de itinerario por correo electrónico y mensaje de texto a los datos de contacto de la reserva. Mantenga sus datos actualizados en «Mis viajes»: si el aviso no llega porque los datos están incorrectos, los derechos de esta política se mantienen, pero no podremos avisarle a tiempo.

## 2. Asistencia en el aeropuerto durante un retraso

Si el retraso respecto de la hora de salida programada es de 2 horas o más, Pacífico Sur entrega un vale de alimentación de CLP 10.000 por pasajero, y otro adicional por cada 3 horas más de espera.

Si el retraso es de 4 horas o más, el pasajero puede desistir del viaje y pedir la devolución completa del pasaje, incluidas las tasas, sin cargo y cualquiera sea la tarifa.

Si la nueva hora de salida obliga a pasar la noche fuera de su ciudad de residencia, Pacífico Sur cubre el hotel y el traslado entre el aeropuerto y el hotel. Si no hay hoteles disponibles, se reembolsa el alojamiento contratado por el pasajero hasta CLP 90.000 por noche, presentando la boleta o factura.

## 3. Cancelaciones

Si Pacífico Sur cancela su vuelo, usted puede elegir una de estas opciones: viajar en el siguiente vuelo con asientos disponibles a su destino, sin costo; cambiar a otra fecha dentro de los 30 días siguientes, sin cargo ni diferencia tarifaria; o pedir la devolución completa del pasaje, incluidas las tasas, cualquiera sea la tarifa.

Las devoluciones por cancelación se solicitan en «Mis viajes» y se pagan según los plazos de la Política de devoluciones vigente.

Si la cancelación se informa con menos de 7 días de anticipación y la causa es atribuible a Pacífico Sur, además de las opciones anteriores el pasajero recibe una compensación de CLP 40.000 en saldo a favor por cada tramo cancelado.

## 4. Reprogramaciones

Si Pacífico Sur cambia el horario de su vuelo en más de 2 horas, usted puede aceptar el nuevo horario, cambiar sin costo a otro vuelo de la misma ruta dentro de los 30 días siguientes, o pedir la devolución completa del pasaje.

## 5. Causas externas

Cuando el retraso o la cancelación se deben a condiciones meteorológicas, cierres del aeropuerto, decisiones de la autoridad aeronáutica u otras causas que no dependen de Pacífico Sur, se mantienen la asistencia en el aeropuerto, el cambio sin costo y la devolución, pero no corresponde la compensación en saldo a favor de la sección 3.

## 6. Conexiones perdidas

Si pierde una conexión por el retraso de un vuelo anterior incluido en la misma reserva, Pacífico Sur lo traslada sin costo en el siguiente vuelo disponible a su destino final y le entrega la asistencia de la sección 2 mientras espera. Las conexiones compradas en reservas separadas no están protegidas.

## 7. Reclamos

Los reclamos por retrasos y cancelaciones se presentan en «Mis viajes» dentro de los 30 días siguientes a la fecha del vuelo, adjuntando las boletas de los gastos que solicita reembolsar.
""",
    },
    {
        "doc_id": "pol-asistencia",
        "title": "Política de asistencia especial",
        "status": "vigente",
        "valid_from": "2026-03-01",
        "valid_to": None,
        "text": """# Política de asistencia especial

Pacífico Sur · Vigente desde el 01/03/2026 · Aplica a pasajeros que necesitan ayuda para viajar.

## 1. Movilidad reducida

Pacífico Sur ofrece sin costo silla de ruedas en el aeropuerto, asistencia para embarcar y desembarcar, y prioridad en el embarque a pasajeros con movilidad reducida.

La asistencia debe solicitarse al menos 48 horas antes de la salida del vuelo en «Mis viajes» o en el centro de contacto. Las solicitudes hechas con menos anticipación se atienden según disponibilidad.

Las sillas de ruedas propias y otros dispositivos de movilidad viajan sin costo en la bodega y no cuentan dentro del equipaje permitido de la tarifa. Las sillas eléctricas deben informarse indicando el tipo de batería.

## 2. Menores de edad que viajan solos

Los menores de 5 años no pueden viajar sin un adulto. Los menores de entre 5 y 13 años que viajen sin un acompañante mayor de 18 años deben usar el servicio de menor no acompañado, que tiene un costo de CLP 35.000 por tramo.

El servicio de menor no acompañado debe contratarse al menos 72 horas antes del vuelo y solo está disponible en vuelos directos nacionales. Un funcionario de Pacífico Sur acompaña al menor desde el control de seguridad hasta la entrega al adulto autorizado en el destino.

Los menores de entre 14 y 17 años pueden viajar solos sin el servicio, presentando la autorización notarial que exija la autoridad para su viaje.

## 3. Embarazo

Las pasajeras embarazadas pueden volar sin restricciones hasta la semana 28 de gestación. Entre las semanas 29 y 35 deben presentar un certificado médico, emitido dentro de los 7 días anteriores al vuelo, que indique las semanas de gestación y que no hay impedimento para volar. Desde la semana 36 no se permite volar.

## 4. Asistencia médica y oxígeno

Los pasajeros que necesiten oxígeno a bordo pueden usar concentradores portátiles aprobados, informándolo al menos 48 horas antes del vuelo y llevando baterías para el 150% de la duración del viaje. Pacífico Sur no provee oxígeno medicinal a bordo, salvo en emergencias.

## 5. Perros de asistencia

Los perros de asistencia de personas con discapacidad viajan en cabina sin costo, junto a su usuario, presentando la credencial o certificado que acredite su función. Deben informarse al momento de solicitar la asistencia.

## 6. Acompañante

Si un pasajero no puede valerse por sí mismo durante el vuelo, por ejemplo para alimentarse o ir al baño, debe viajar con un acompañante mayor de 18 años. La tripulación no puede prestar ese tipo de asistencia personal.
""",
    },
    {
        "doc_id": "pol-embarque",
        "title": "Política de check-in y embarque",
        "status": "vigente",
        "valid_from": "2026-03-01",
        "valid_to": None,
        "text": """# Política de check-in y embarque

Pacífico Sur · Vigente desde el 01/03/2026 · Aplica a todos los vuelos operados por Pacífico Sur.

## 1. Check-in

El check-in en línea, en el sitio web o en la aplicación, se abre 48 horas antes de la salida del vuelo y cierra 2 horas antes en vuelos nacionales y 3 horas antes en vuelos internacionales.

El check-in en el aeropuerto es obligatorio para quienes facturan equipaje, viajan con un servicio de asistencia especial o con un menor no acompañado. Los mesones cierran 1 hora antes de la salida en vuelos nacionales y 2 horas antes en vuelos internacionales.

El check-in en el mesón del aeropuerto es gratuito para todas las tarifas. Imprimir la tarjeta de embarque en el aeropuerto tiene un costo de CLP 5.000 en tarifa Light.

## 2. Documentos de viaje

En vuelos nacionales se debe presentar la cédula de identidad o el pasaporte vigente. En vuelos internacionales, el pasaporte o el documento que exija el país de destino, además de las visas y autorizaciones que correspondan. Revisar los requisitos de entrada es responsabilidad del pasajero.

## 3. Embarque

La puerta de embarque abre 45 minutos antes de la salida. El embarque cierra 20 minutos antes de la salida en vuelos nacionales y 30 minutos antes en vuelos internacionales; después de ese momento no se permite subir al avión.

El embarque se hace por grupos: primero los pasajeros con asistencia especial y familias con niños menores de 2 años, después la cabina Premium y los pasajeros Full, y luego el resto según el grupo impreso en la tarjeta de embarque.

## 4. Pasajero no presentado

Si no se presenta a un vuelo sin haberlo cambiado antes, el pasaje de ese tramo se considera usado. Si no se presenta al vuelo de ida, Pacífico Sur cancela automáticamente los tramos siguientes de la misma reserva, incluido el vuelo de vuelta.

Para conservar el vuelo de vuelta, avise en «Mis viajes» o en el centro de contacto antes de la salida del vuelo de ida que no lo utilizará.

## 5. Asientos

Los pasajes Light reciben un asiento asignado al azar en el check-in; elegir un asiento antes cuesta desde CLP 4.000 por tramo. Los pasajes Plus incluyen la elección de asientos estándar y los Full, la de cualquier asiento, incluidos los de salida de emergencia.

Los asientos de salida de emergencia solo se asignan a pasajeros mayores de 15 años, que puedan ayudar en una evacuación y que no viajen con asistencia especial.

## 6. Dispositivos electrónicos a bordo

Los teléfonos, tablets y computadores pueden usarse durante todo el vuelo en modo avión. Las baterías externas deben viajar en el equipaje de mano y no pueden usarse para cargar dispositivos durante el vuelo.
""",
    },
]

# Cada pregunta: texto, respuesta de referencia y evidencia esperada (documento y frase literal).
# Las preguntas sin respuesta llevan evidencia vacía y la respuesta «sin evidencia».
QUESTIONS = [
    ("q01", "Viajo en Light. ¿Puedo subir al avión con mi maleta chica de rueditas?",
     "Solo un artículo personal de hasta 3 kg que quepa bajo el asiento; la maleta de mano se agrega antes del vuelo en «Mis viajes» por CLP 12.000 por tramo, o se cobra CLP 35.000 si se detecta en la puerta.",
     [("pol-equipaje", "La tarifa Light incluye solo el artículo personal."),
      ("pol-equipaje", "de hasta 3 kg y que quepa bajo el asiento delantero"),
      ("pol-equipaje", "debe agregarla antes del vuelo en «Mis viajes» por CLP 12.000 por tramo"),
      ("pol-equipaje", "se cobra CLP 35.000 por tramo y se envía a la bodega")],
     "un documento"),
    ("q02", "Mi valija no aparece hace tres semanas. ¿Ya la consideran extraviada?",
     "Cuando no se ubica dentro de 21 días corridos desde la fecha del vuelo.",
     [("pol-equipaje", "Una maleta demorada se considera perdida si no se ubica dentro de 21 días corridos desde la fecha del vuelo.")],
     "un documento"),
    ("q03", "Se me olvidó pagar la valija grande y ya estoy en el mesón. ¿Cuánto me van a cobrar?",
     "CLP 40.000 por tramo; en «Mis viajes», hasta 4 horas antes, cuesta CLP 22.000.",
     [("pol-equipaje", "CLP 40.000 por tramo si se compra en el aeropuerto")],
     "un documento"),
    ("q04", "Me entregaron la valija con una rueda quebrada y me di cuenta en la casa. ¿Todavía puedo reclamar?",
     "Hasta 7 días corridos desde la entrega de la maleta, con fotos del daño y la etiqueta de equipaje.",
     [("pol-equipaje", "a más tardar, dentro de 7 días corridos desde la entrega de la maleta")],
     "un documento"),
    ("q05", "Con mi pasaje Light necesito viajar una semana más tarde. ¿Qué me cobran?",
     "Un cambio de fecha por tramo, con un cargo de CLP 25.000 más la diferencia tarifaria, hecho hasta 3 horas antes de la salida del vuelo original.",
     [("pol-cambios", "Los pasajes Light admiten un cambio de fecha u hora por tramo, con un cargo de CLP 25.000 más la diferencia tarifaria."),
      ("pol-cambios", "Todos los cambios deben hacerse hasta 3 horas antes de la salida del vuelo original")],
     "un documento"),
    ("q06", "Terminé antes la reunión y quiero irme en el vuelo de las 15 en vez del de las 20. Tengo Plus. ¿Cuánto pago?",
     "CLP 10.000, sin diferencia tarifaria, según disponibilidad y en la misma ruta.",
     [("pol-cambios", "El cambio el mismo día cuesta CLP 10.000 en Plus y no tiene costo en Full, sin cobrar diferencia tarifaria.")],
     "distractor"),
    ("q07", "Mi pareja no puede viajar. ¿Puedo poner su pasaje a nombre de mi hermano?",
     "No, los pasajes no se transfieren; solo se corrige sin costo un error de hasta tres letras en el nombre.",
     [("pol-cambios", "No se permite transferir un pasaje a otra persona.")],
     "un documento"),
    ("q08", "¿Puedo viajar con mi gato en la cabina?",
     "sin evidencia",
     [],
     "sin respuesta"),
    ("q09", "Compré el pasaje en marzo y no lo usé. ¿Todavía alcanzo a pedir que me devuelvan la plata?",
     "Hasta un año después de la fecha de emisión del pasaje; pasado ese plazo solo se devuelven las tasas de embarque e impuestos.",
     [("pol-devoluciones-2026", "Puede solicitar la devolución hasta un año después de la fecha de emisión del pasaje."),
      ("pol-devoluciones-2026", "Pasado ese plazo, solo se devuelven las tasas de embarque e impuestos.")],
     "versión"),
    ("q10", "Saqué la tarifa más barata, la Light, y al final no voy. ¿Recupero algo?",
     "La tarifa no, pero sí las tasas de embarque e impuestos, que se devuelven siempre.",
     [("pol-devoluciones-2026", "Los pasajes Light no tienen devolución de la tarifa."),
      ("pol-devoluciones-2026", "Las tasas de embarque e impuestos se devuelven siempre, en todas las tarifas")],
     "versión"),
    ("q11", "Esta mañana compré en la app un vuelo para el próximo mes y me arrepentí. ¿Me cobran algo por anularlo?",
     "Sí, si el vuelo sale al menos 7 días después de la compra: el retracto dentro de 24 horas da devolución completa sin cargo, incluso en Light.",
     [("pol-devoluciones-2026", "dentro de 24 horas desde la compra, con devolución completa y sin cargo, siempre que el vuelo salga al menos 7 días después de la compra")],
     "versión"),
    ("q12", "Llevo tres horas esperando en la puerta y el avión no sale. ¿Me tienen que dar algo para comer?",
     "Un vale de alimentación de CLP 10.000, porque el retraso es de 2 horas o más.",
     [("pol-retrasos", "Si el retraso respecto de la hora de salida programada es de 2 horas o más, Pacífico Sur entrega un vale de alimentación de CLP 10.000 por pasajero")],
     "un documento"),
    ("q13", "La aerolínea suspendió mi vuelo de mañana. ¿Qué puedo hacer?",
     "Viajar en el siguiente vuelo disponible sin costo, cambiar a otra fecha dentro de 30 días sin cargo, o pedir la devolución completa con tasas. Si se avisó con menos de 7 días y la causa es de la aerolínea, además recibe CLP 40.000 en saldo a favor por tramo.",
     [("pol-retrasos", "viajar en el siguiente vuelo con asientos disponibles a su destino, sin costo; cambiar a otra fecha dentro de los 30 días siguientes, sin cargo ni diferencia tarifaria; o pedir la devolución completa del pasaje, incluidas las tasas, cualquiera sea la tarifa"),
      ("pol-retrasos", "recibe una compensación de CLP 40.000 en saldo a favor por cada tramo cancelado")],
     "un documento"),
    ("q14", "Por el atraso tengo que dormir en Punta Arenas y no vivo ahí. ¿Quién paga la noche?",
     "Pacífico Sur cubre el hotel y el traslado; si no hay hoteles, reembolsa hasta CLP 90.000 por noche con boleta, reclamando en «Mis viajes» dentro de 30 días.",
     [("pol-retrasos", "Pacífico Sur cubre el hotel y el traslado entre el aeropuerto y el hotel"),
      ("pol-retrasos", "se reembolsa el alojamiento contratado por el pasajero hasta CLP 90.000 por noche"),
      ("pol-retrasos", "Los reclamos por retrasos y cancelaciones se presentan en «Mis viajes» dentro de los 30 días siguientes a la fecha del vuelo")],
     "un documento"),
    ("q15", "¿Puedo ver series conectado a internet durante el vuelo?",
     "sin evidencia",
     [],
     "sin respuesta"),
    ("q16", "¿Puede mi hija de 10 años viajar sola de Santiago a Calama?",
     "Sí, con el servicio de menor no acompañado: CLP 35.000 por tramo, contratado al menos 72 horas antes y solo en vuelos directos nacionales; el check-in debe hacerse en el aeropuerto.",
     [("pol-asistencia", "Los menores de entre 5 y 13 años que viajen sin un acompañante mayor de 18 años deben usar el servicio de menor no acompañado, que tiene un costo de CLP 35.000 por tramo."),
      ("pol-asistencia", "debe contratarse al menos 72 horas antes del vuelo y solo está disponible en vuelos directos nacionales"),
      ("pol-embarque", "El check-in en el aeropuerto es obligatorio para quienes facturan equipaje, viajan con un servicio de asistencia especial o con un menor no acompañado.")],
     "un documento"),
    ("q17", "Mi vuelo se retrasó 5 horas y ya no quiero viajar. ¿Me devuelven todo y en cuánto tiempo?",
     "Sí, la devolución completa con tasas y sin cargo, en cualquier tarifa; una vez aprobada se paga en 10 días hábiles.",
     [("pol-retrasos", "Si el retraso es de 4 horas o más, el pasajero puede desistir del viaje y pedir la devolución completa del pasaje, incluidas las tasas, sin cargo y cualquiera sea la tarifa."),
      ("pol-devoluciones-2026", "Una vez aprobada la solicitud, la devolución se paga en un plazo de 10 días hábiles.")],
     "dos documentos"),
    ("q18", "A mi papá le cuesta caminar distancias largas. ¿Qué tengo que hacer para que lo lleven en silla en el aeropuerto?",
     "Al menos 48 horas antes de la salida; con menos anticipación se atiende según disponibilidad. No tiene costo.",
     [("pol-asistencia", "La asistencia debe solicitarse al menos 48 horas antes de la salida del vuelo")],
     "un documento"),
    ("q19", "Tengo 32 semanas de embarazo. ¿Me dejan volar?",
     "Sí, con un certificado médico emitido dentro de los 7 días anteriores al vuelo; desde la semana 36 no se permite volar.",
     [("pol-asistencia", "Entre las semanas 29 y 35 deben presentar un certificado médico, emitido dentro de los 7 días anteriores al vuelo")],
     "un documento"),
    ("q20", "Si llego corriendo a la puerta 15 minutos antes de mi vuelo a Antofagasta, ¿me dejan subir?",
     "No: en vuelos nacionales el embarque cierra 20 minutos antes de la salida.",
     [("pol-embarque", "El embarque cierra 20 minutos antes de la salida en vuelos nacionales")],
     "distractor"),
    ("q21", "¿Puedo canjear mis millas para pasarme a Premium?",
     "sin evidencia",
     [],
     "sin respuesta"),
    ("q22", "No alcancé a tomar el vuelo de ida. ¿Pierdo también la vuelta?",
     "Sí: si no se presenta a la ida, se cancelan los tramos siguientes. Para conservar la vuelta hay que avisar en «Mis viajes» o en el centro de contacto antes de la salida de la ida.",
     [("pol-embarque", "Si no se presenta al vuelo de ida, Pacífico Sur cancela automáticamente los tramos siguientes de la misma reserva, incluido el vuelo de vuelta."),
      ("pol-embarque", "Para conservar el vuelo de vuelta, avise en «Mis viajes» o en el centro de contacto antes de la salida del vuelo de ida que no lo utilizará.")],
     "un documento"),
    ("q23", "Viajo a Lima el jueves. ¿Desde cuándo y hasta cuándo puedo sacar la tarjeta de embarque por internet?",
     "Desde 48 horas antes de la salida hasta 3 horas antes.",
     [("pol-embarque", "se abre 48 horas antes de la salida del vuelo y cierra 2 horas antes en vuelos nacionales y 3 horas antes en vuelos internacionales")],
     "distractor"),
    ("q24", "Ya me aprobaron la devolución. ¿En cuánto tiempo me pagan?",
     "En un plazo de 10 días hábiles desde la aprobación, al mismo medio de pago de la compra.",
     [("pol-devoluciones-2026", "Una vez aprobada la solicitud, la devolución se paga en un plazo de 10 días hábiles."),
      ("pol-devoluciones-2026", "El pago se hace al mismo medio de pago usado en la compra")],
     "versión"),
    # Agregadas el 28/10/2026: la evidencia llega, pero hay que comprobar una condición numérica que no se cumple.
    # Gemini 3.5 Flash Lite respondió «sí» en 4 de 4 ejecuciones de cada una, con citas literales correctas.
    ("q25", "Ayer compré por la app un pasaje Light para un vuelo que sale en 3 días. Me arrepentí, ¿me devuelven todo?",
     "No: el retracto exige que el vuelo salga al menos 7 días después de la compra. Con Light no se devuelve la tarifa; sí las tasas de embarque e impuestos.",
     [("pol-devoluciones-2026", "siempre que el vuelo salga al menos 7 días después de la compra"),
      ("pol-devoluciones-2026", "Los pasajes Light no tienen devolución de la tarifa."),
      ("pol-devoluciones-2026", "Las tasas de embarque e impuestos se devuelven siempre")],
     "condición"),
    ("q26", "Mi vuelo lleva 3 horas de atraso y ya no quiero viajar. ¿Me devuelven todo el pasaje?",
     "No por el retraso: la devolución completa por desistir exige un retraso de 4 horas o más.",
     [("pol-retrasos", "Si el retraso es de 4 horas o más, el pasajero puede desistir del viaje y pedir la devolución completa del pasaje")],
     "condición"),
]

# Subconjunto de 12 preguntas para generar en clase: las tres sin respuesta,
# las dos de la demostración (q17 y q24), una de cada tipo restante y las dos de condición (q25 y q26).
GENERATION_SUBSET = ["q02", "q08", "q09", "q10", "q13", "q15", "q17", "q19", "q21", "q24", "q25", "q26"]


def main() -> None:
    docs_dir = DEST / "documents"
    docs_dir.mkdir(parents=True, exist_ok=True)
    texts = {}
    manifest = []
    for doc in DOCUMENTS:
        filename = f"{doc['doc_id']}.md"
        (docs_dir / filename).write_text(doc["text"], encoding="utf-8")
        texts[doc["doc_id"]] = doc["text"]
        manifest.append({k: doc[k] for k in ("doc_id", "title", "status", "valid_from", "valid_to")} | {"file": filename})

    questions = []
    for qid, question, answer, evidence, category in QUESTIONS:
        for doc_id, quote in evidence:
            if quote not in texts[doc_id]:
                raise SystemExit(f"{qid}: la frase esperada no aparece en {doc_id}: {quote!r}")
        questions.append({
            "id": qid,
            "question": question,
            "expected_answer": answer,
            "expected_evidence": [{"doc_id": d, "quote": q} for d, q in evidence],
            "answerable": bool(evidence),
            "category": category,
            "in_generation_subset": qid in GENERATION_SUBSET,
        })

    (DEST / "documents.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (DEST / "questions.json").write_text(json.dumps(questions, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"{len(manifest)} documentos y {len(questions)} preguntas ({sum(q['answerable'] for q in questions)} con respuesta) en {DEST}")


if __name__ == "__main__":
    main()
