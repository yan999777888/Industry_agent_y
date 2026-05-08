"""Lightweight customer-service policy responses for non-manual questions."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class PolicyResponse:
    answer: str
    confidence: float
    matched_topics: list[str]


@dataclass(frozen=True)
class ScenarioRule:
    name: str
    terms: tuple[str, ...]
    overview: str = ""
    materials: str = ""
    timeline: str = ""
    fees: str = ""
    eligibility: str = ""
    process: str = ""
    contact: str = ""


@dataclass(frozen=True)
class TopicRule:
    topic: str
    terms: tuple[str, ...]
    overview: str
    materials: str
    timeline: str
    fees: str
    eligibility: str
    process: str
    contact: str
    scenarios: tuple[ScenarioRule, ...] = field(default_factory=tuple)


_TOPIC_RULES: tuple[TopicRule, ...] = (
    TopicRule(
        topic="refund_exchange",
        terms=("退货", "换货", "退换货", "退款"),
        overview="退换货和退款通常要结合订单状态、签收时间、商品是否使用以及平台规则综合判断。",
        materials="建议先准备订单号、购买渠道、签收时间、商品当前状态，以及商品照片或开箱视频。",
        timeline="如果符合条件，退款到账时间通常还要看支付方式和平台审核进度，实际时效以原支付渠道处理结果为准。",
        fees="是否需要承担运费，通常取决于退换原因、商品是否存在质量问题以及平台售后规则。",
        eligibility="若商品未使用、仍在退换时效内，通常更容易申请退货、换货或退款；是否最终通过仍要以售后审核为准。",
        process="一般建议先在订单页发起售后申请，并同步说明问题原因、商品状态和诉求，必要时上传图片或视频证据。",
        contact="如果页面无法直接发起售后，建议携带订单号和证据材料联系人工客服进一步处理。",
        scenarios=(
            ScenarioRule(
                name="seven_day_no_reason",
                terms=("7天无理由", "七天无理由", "不想要了", "不喜欢", "不合适", "买错了"),
                overview="如果是未使用、配件齐全且仍在 7 天无理由范围内，通常更适合按无理由退货处理。",
                materials="通常需要订单号、签收时间，以及商品未使用、外包装和配件完整的说明或照片。",
                timeline="无理由退货的退款时效通常取决于退货寄回、仓库签收和平台审核进度。",
                fees="无质量问题的无理由退货，运费通常更可能由买家承担；是否支持运费险抵扣要看平台规则。",
                eligibility="是否能走 7 天无理由，通常取决于商品类目、是否拆封使用以及是否仍在时效内。",
                process="建议先在订单页选择无理由退货，并说明商品未使用、配件齐全的状态。",
                contact="如果页面提示该商品不支持无理由退货，建议带上订单号联系人工客服核实类目限制。",
            ),
            ScenarioRule(
                name="quality_reason",
                terms=("质量问题", "有问题", "故障", "坏了", "破损", "瑕疵"),
                overview="如果退货诉求是因为质量问题，更适合按质量售后或故障退换处理，而不是普通无理由退货。",
                materials="建议准备故障照片、视频、异常描述以及签收时间，最好保留开箱证据。",
                timeline="质量问题的审核和退款时效通常会受售后判责、检测流程和退款路径影响。",
                fees="若经核实属于质量问题，运费和退换责任通常更可能由商家或平台承担。",
                eligibility="是否支持直接退款或换货，通常取决于问题严重程度、签收时间和证据完整度。",
                process="建议先提交质量问题售后申请，并尽量上传清晰的故障照片、视频或开箱记录。",
                contact="如果页面无法上传证据或判责有争议，建议尽快联系人工客服升级处理。",
            ),
            ScenarioRule(
                name="opened_or_used",
                terms=("拆封", "开封", "使用过", "激活", "安装过", "试用过"),
                overview="如果商品已经拆封、激活或安装使用，通常更需要结合平台规则和售后审核判断是否还能退换。",
                materials="建议准备订单号、当前商品状态、拆封或使用情况说明，以及是否还能恢复完整包装的照片。",
                timeline="已拆封商品的审核时效通常取决于卖家判定和是否需要进一步检测。",
                fees="已拆封或已使用商品是否承担运费，通常更依赖具体类目规则和售后审核结果。",
                eligibility="是否还能退换，通常要看是否影响二次销售、是否存在质量问题以及平台规则。",
                process="建议先如实说明拆封和使用情况，再提交售后申请，避免因信息不一致影响审核。",
                contact="如果你不确定当前状态是否还能退换，建议带上订单号联系人工客服先做规则确认。",
            ),
            ScenarioRule(
                name="refund_rejected",
                terms=("驳回", "拒绝退款", "不通过", "审核失败", "没通过"),
                overview="如果退款或退货申请已经被驳回，通常需要先看驳回原因，再决定是补证据、改诉求还是升级申诉。",
                materials="建议准备订单号、售后单截图、驳回原因说明，以及商品当前状态、聊天记录或问题证据。",
                timeline="被驳回后的再次审核时效通常取决于补充材料是否完整以及平台工单处理进度。",
                fees="是否仍涉及运费或退款扣减，通常要结合驳回原因、商品状态和平台规则确认。",
                eligibility="是否还能重新提交，通常取决于驳回原因、是否仍在售后时效内以及是否有新的证据补充。",
                process="建议先查看驳回原因；如果是材料不足，就补齐证据重新提交；如果你认为判定不合理，可发起申诉或联系客服升级。",
                contact="如果页面没有再次提交入口，建议带上售后单截图联系人工客服复核。",
            ),
        ),
    ),
    TopicRule(
        topic="invoice",
        terms=("发票", "开发票", "抬头"),
        overview="发票问题通常需要确认订单号、开票类型、抬头信息以及当前开票状态。",
        materials="建议先准备订单号、开票类型、发票抬头、税号和接收邮箱等信息。",
        timeline="开票和补开发票的处理时效通常与订单状态、开票系统和平台规则有关，实际时间以平台处理进度为准。",
        fees="发票本身是否收费通常要看平台政策；如果涉及重开或邮寄，是否额外收费也要以平台规则为准。",
        eligibility="是否支持重开或修改抬头，通常取决于发票是否已开具以及平台是否允许更改。",
        process="一般建议先确认订单是否满足开票条件，再提交或修改开票信息，必要时联系人工客服协助处理。",
        contact="如果你不确定发票状态，建议携带订单号直接联系人工客服核实。",
        scenarios=(
            ScenarioRule(
                name="invoice_reissue",
                terms=("重开", "重开发票", "开错", "抬头填错", "税号错", "邮箱错"),
                overview="如果发票信息填错或已经开错，通常要先确认发票是否已开具、平台是否支持重开以及错误项属于哪一类。",
                materials="建议准备订单号、错误发票截图、正确的抬头/税号/邮箱信息，以及原申请记录。",
                timeline="发票重开的处理时效通常要看原发票状态、财务流程和平台规则，实际进度以系统审核为准。",
                fees="是否会产生邮寄费或重开成本，通常要结合平台规则和当前发票形态确认。",
                eligibility="是否支持重开，通常取决于发票是否已经开具、是否已报销以及平台是否允许修改该字段。",
                process="建议先确认错误项属于抬头、税号还是邮箱；若支持修改或重开，尽快在订单页或客服渠道提交更正申请。",
                contact="如果你不确定当前发票是否还能重开，建议带上订单号和错误发票截图联系人工客服核实。",
            ),
        ),
    ),
    TopicRule(
        topic="shipping",
        terms=("物流", "快递", "发货", "补发", "签收", "运费", "乡镇", "国外", "配送"),
        overview="物流和配送问题通常需要结合订单号、物流单号、收货地址以及当前物流状态判断。",
        materials="建议先准备订单号、物流单号、收货地址、异常时间点以及相关截图。",
        timeline="物流时效通常受仓库出库、承运商揽收和目的地配送能力影响，实际到达时间以物流轨迹为准。",
        fees="是否产生额外运费，通常要看配送地区、补发原因和物流方案，需结合订单系统进一步确认。",
        eligibility="乡镇、海外或特殊地区配送是否支持，通常要看平台覆盖范围和商品限制。",
        process="建议先核对物流轨迹、签收状态和地址信息；若长期停滞、少件或误签，可尽快发起物流异常反馈。",
        contact="如物流长时间无更新或疑似丢件，建议携带订单号和物流单号联系人工客服或承运商处理。",
        scenarios=(
            ScenarioRule(
                name="tracking_stalled",
                terms=("没更新", "不更新", "没动", "停滞", "卡住", "未揽收", "一直没有物流"),
                overview="如果物流长时间没有更新，更像是揽收延迟、中转停滞或系统轨迹未同步的场景。",
                materials="建议准备订单号、物流单号、最近一次轨迹时间以及页面截图。",
                timeline="这类物流停滞的处理时效通常要看仓库、承运商反馈以及是否需要人工催查。",
                fees="物流停滞本身通常不是收费问题，重点是先确认是否超过承诺时效。",
                eligibility="是否能按异常物流处理，通常要对照最近轨迹时间和平台承诺时效判断。",
                process="建议先核对最近一次轨迹更新时间；如果明显超时，可提交物流催查或人工催件。",
                contact="如果轨迹超过较长时间仍无变化，建议同时联系人工客服和承运商催查。",
            ),
            ScenarioRule(
                name="signed_but_missing",
                terms=("已签收", "显示签收", "代签", "没收到", "未收到"),
                overview="如果物流显示已签收但实际未收到，更像是代签、误投或末端配送异常场景。",
                materials="建议准备订单号、物流单号、签收时间、签收截图，以及门卫、驿站或家人是否代收的信息。",
                timeline="已签收未收到通常需要先核实末端签收情况，处理时效取决于承运商回查速度。",
                fees="这类问题通常不以费用为主，重点是先确认签收责任和包裹去向。",
                eligibility="是否可以按丢件或误签处理，通常要看签收记录、代签情况和回查结果。",
                process="建议先联系承运商核实签收人，再同步在平台提交物流异常或未收到申请。",
                contact="如果承运商和平台说法不一致，建议带上轨迹截图联系人工客服升级处理。",
            ),
            ScenarioRule(
                name="redirect_or_wrong_address",
                terms=("改派", "拦截", "送错", "发错地址", "派错", "改地址"),
                overview="如果是改派、拦截或送错地址，通常要先确认订单地址、物流状态以及末端站点是否支持改派。",
                materials="建议准备订单号、物流单号、原地址、新地址和当前轨迹截图。",
                timeline="地址改派或拦截的处理时效通常取决于包裹是否已进入末端配送以及承运商是否支持操作。",
                fees="是否收费通常要看改派距离、承运商政策和是否需要二次派送。",
                eligibility="是否还能改派，通常取决于包裹是否已签收、是否到站以及承运商规则。",
                process="建议先联系承运商确认是否支持改派，再在平台侧同步更新或备注地址异常。",
                contact="如果系统侧和物流侧信息不一致，建议携带订单号联系人工客服协调处理。",
            ),
        ),
    ),
    TopicRule(
        topic="complaint",
        terms=("投诉", "虚假宣传", "假货", "二手", "辱骂", "赔偿"),
        overview="投诉、假货、虚假宣传、二手商品或服务态度问题通常需要先固定证据再升级处理。",
        materials="建议准备订单记录、聊天记录、商品照片、视频以及页面宣传截图等证据。",
        timeline="投诉和升级处理的时效通常取决于平台工单流程和证据完整度，实际进度以平台反馈为准。",
        fees="投诉本身是否涉及额外费用通常不是重点，更重要的是先固定证据并确认赔付依据。",
        eligibility="是否支持赔偿或升级处理，通常要看证据是否充分以及平台判责结果。",
        process="建议优先通过平台投诉单、售后工单或人工客服渠道提交证据并说明诉求。",
        contact="如果问题较严重，建议直接联系人工客服并走官方投诉流程。",
    ),
    TopicRule(
        topic="after_sales",
        terms=("售后", "维修", "保修", "保修卡", "保障卡", "维修费用"),
        overview="售后、维修和保修问题通常需要确认商品型号、故障现象、购买时间和保修凭证。",
        materials="建议准备订单号、商品型号、故障描述、故障照片或视频以及保修凭证。",
        timeline="维修和保修审核时效通常要看检测流程、备件情况和售后网点安排，实际时间以售后反馈为准。",
        fees="是否收费通常取决于是否人为损坏、是否在保修期内以及具体维修项目。",
        eligibility="若商品仍在保修期且非人为损坏，通常更容易进入保修流程；最终结果仍以售后检测为准。",
        process="建议先提交售后申请并说明故障现象，必要时配合寄修、检测或预约售后网点。",
        contact="如果你不确定是否在保修范围内，建议携带凭证联系人工客服或官方售后确认。",
        scenarios=(
            ScenarioRule(
                name="in_warranty",
                terms=("在保", "保修期内", "没过保", "还在保修期"),
                overview="如果确认仍在保修期内，通常更适合先按保修检测流程推进。",
                materials="建议准备订单号、购买时间、保修凭证、型号信息和故障照片或视频。",
                timeline="在保维修的处理时效通常取决于检测排队、配件库存和网点安排。",
                fees="如果检测后确认为非人为损坏且仍在保修范围内，通常更可能免维修费；运费是否减免仍要看政策。",
                eligibility="是否最终按保修处理，还要看故障原因、购买时间和检测结论是否一致。",
                process="建议先提交保修申请，并在描述里明确购买时间、故障现象和是否仍在保修期内。",
                contact="如果你已经有保修凭证，建议直接联系官方售后或人工客服加快核验。",
            ),
            ScenarioRule(
                name="out_of_warranty",
                terms=("过保", "超出保修期", "保修过了", "过了保修期"),
                overview="如果已经过保，通常更可能进入收费维修或付费检测流程，而不是标准保修。",
                materials="建议准备订单号、商品型号、购买时间和当前故障情况说明。",
                timeline="过保维修的处理时效通常取决于检测、报价确认和备件供应情况。",
                fees="过保后通常更可能产生检测费、维修费或配件费，具体仍要以售后报价为准。",
                eligibility="是否还能获得免费处理，通常只有在特殊活动、延保或明确质量责任场景下才更有可能。",
                process="建议先咨询售后是否支持付费检测，再确认报价和是否值得维修。",
                contact="如果你拿不准是否真的过保，建议带上订单号和购买时间联系人工客服先核验。",
            ),
            ScenarioRule(
                name="human_damage",
                terms=("人为", "进水", "摔坏", "磕碰", "私拆", "外力"),
                overview="如果故障涉及进水、摔落、磕碰或私拆，更像是人为损坏场景，通常会影响是否能走保修。",
                materials="建议准备故障照片、受损部位说明、购买时间和当前是否还能开机的情况。",
                timeline="人为损坏的处理时效通常取决于检测、责任认定和维修报价确认。",
                fees="如果最终判定为人为损坏，通常更可能产生检测费或维修费。",
                eligibility="这类情况是否还能免费保修，通常要以售后检测和品牌规则为准，很多场景下更可能进入付费维修。",
                process="建议先提交检测或维修申请，并如实说明进水、跌落或私拆等情况，避免后续判责反复。",
                contact="如果你担心被误判，建议在寄修前带上照片和订单号联系人工客服或官方售后做预判。",
            ),
        ),
    ),
    TopicRule(
        topic="quality_issue",
        terms=("划痕", "破损", "瑕疵", "少件", "少了一件", "保质期"),
        overview="破损、瑕疵、少件或保质期异常这类问题，通常需要尽快留证并核对签收时间。",
        materials="建议准备订单号、签收时间、问题照片、开箱视频以及异常细节说明。",
        timeline="若问题发生在签收后较短时间内，通常更容易进入补发、换货或售后处理流程。",
        fees="是否产生额外费用通常要看问题原因和平台售后规则，质量问题一般更适合优先申请售后判责。",
        eligibility="是否支持补发、换货或退款，通常要看问题是否属于运输或质量异常，以及证据是否充分。",
        process="建议先在订单页发起售后，并同步上传图片或视频说明问题。",
        contact="如系统无法提交售后，建议直接联系人工客服升级处理。",
    ),
    TopicRule(
        topic="order_change",
        terms=("取消订单", "订单", "到账", "信用卡", "原路返回"),
        overview="订单取消和退款到账问题通常取决于订单是否发货、支付方式以及平台规则。",
        materials="建议准备订单号、支付方式、支付时间和相关账单截图。",
        timeline="退款一般按原支付路径退回，到账时间通常受支付渠道和平台审核流程影响。",
        fees="是否产生手续费通常要看支付渠道和订单场景，具体仍要以支付渠道规则为准。",
        eligibility="如果订单尚未发货，通常更容易申请取消；已发货订单则可能需要走拒收或售后流程。",
        process="建议先确认订单状态，再在订单页提交取消或售后申请。",
        contact="若页面状态异常，建议携带订单号联系人工客服核实。",
    ),
    TopicRule(
        topic="manual_request",
        terms=("纸质版说明书", "电子版", "说明书"),
        overview="说明书补发或获取电子版通常需要先确认具体商品名称和型号。",
        materials="建议准备商品名称、型号、订单号以及需要的说明书版本。",
        timeline="电子版通常更容易即时获取；纸质版是否支持补寄以及时效，通常要看包装配置和售后政策。",
        fees="纸质版补寄是否收费通常要以商品售后规则和客服核实结果为准。",
        eligibility="是否支持补寄纸质说明书，通常要看商品原始包装清单和品牌政策。",
        process="建议优先通过商品页面、品牌官网或客服渠道获取电子版，再确认是否支持纸质补寄。",
        contact="如果页面找不到说明书入口，建议携带型号联系人工客服。",
    ),
    TopicRule(
        topic="address_change",
        terms=("改地址", "修改地址", "收货地址", "改收件地址", "改配送地址"),
        overview="修改收货地址通常要先看订单是否已经发货或进入出库流程。",
        materials="建议准备订单号、当前订单状态和要修改的新地址。",
        timeline="如果订单还未出库，通常更容易修改；若已发货，时效一般要看物流拦截或改派结果。",
        fees="是否产生额外费用通常要看改派方式、配送地区和平台规则。",
        eligibility="是否还能修改地址，通常取决于订单是否已出库、已发货以及物流是否支持改派。",
        process="建议先确认订单状态；若仍可修改，尽快在订单页或客服渠道提交地址变更申请。",
        contact="若系统内无法直接改地址，建议立即联系人工客服或物流方协助处理。",
    ),
    TopicRule(
        topic="installation_service",
        terms=("预约安装", "上门安装", "安装服务", "师傅安装", "安装预约"),
        overview="预约安装或上门安装服务通常要结合订单状态、商品型号、安装地址和服务覆盖范围判断。",
        materials="建议准备订单号、商品型号、安装地址、联系电话和可预约时间。",
        timeline="预约安装时效通常受地区覆盖和师傅排期影响，最早可预约时间要以系统排期为准。",
        fees="是否收费通常要看商品是否含安装服务、所在地区以及安装项目范围。",
        eligibility="是否支持上门安装，通常要看商品类目、服务覆盖地区和订单状态。",
        process="建议先确认订单是否支持安装服务，再提交预约申请并填写地址与时间。",
        contact="如系统内没有安装入口，建议带上订单号联系人工客服确认。",
        scenarios=(
            ScenarioRule(
                name="installation_reschedule",
                terms=("改约", "改时间", "重新预约", "约不上", "师傅没来", "爽约", "改安装时间"),
                overview="如果已经预约安装但需要改约，或师傅未按约到场，通常要先确认原预约状态和最近可改约档期。",
                materials="建议准备订单号、预约时间、安装地址、联系电话，以及系统预约截图或师傅联系记录。",
                timeline="改约或重新排期的时效通常取决于地区覆盖、师傅排班和最近可预约时间段。",
                fees="是否产生额外费用，通常要看改约原因、是否多次爽约以及平台安装服务规则。",
                eligibility="是否支持改约，通常取决于服务单状态、地区排期和是否已经上门。",
                process="建议先在安装服务单里查看是否支持改约；若系统无法修改，可联系平台或安装师傅重新确认时间。",
                contact="如果师傅爽约或多次改期，建议带上服务单截图联系人工客服升级协调。",
            ),
        ),
    ),
    TopicRule(
        topic="price_protection",
        terms=("保价", "价保", "降价", "买贵了"),
        overview="价保或降价补差通常要结合订单时间、活动规则和平台价保政策判断。",
        materials="建议准备订单号、下单时间、当前商品页面价格截图以及活动页面信息。",
        timeline="是否在价保周期内通常是关键，具体审核时间仍要以平台处理进度为准。",
        fees="价保通常不是收费问题，关键在于是否满足平台补差规则。",
        eligibility="是否支持补差，通常取决于商品是否参加价保、下单时间是否在周期内以及活动是否适用。",
        process="建议先核对商品页面的价保规则，再提交价保申请或联系人工客服核实。",
        contact="如果你不确定是否在价保周期内，建议携带订单号和截图联系人工客服。",
    ),
    TopicRule(
        topic="payment_issue",
        terms=("支付失败", "付款失败", "扣款", "重复扣款", "支付异常", "扣了两次"),
        overview="支付异常问题通常需要先核对订单状态、支付流水和扣款时间。",
        materials="建议准备订单号、支付方式、扣款时间、银行或平台账单截图。",
        timeline="支付异常的处理时效通常取决于支付渠道回执、平台对账和财务核查进度。",
        fees="若涉及重复扣款或资金冻结，是否产生额外费用通常要以支付渠道处理结果为准。",
        eligibility="是否属于支付异常，通常要结合订单是否生成、是否重复扣款和账单记录综合判断。",
        process="建议先确认订单是否生成，再核对支付流水；若账单异常，尽快联系客服发起核查。",
        contact="如资金问题较紧急，建议同时联系平台客服和支付渠道客服处理。",
    ),
    TopicRule(
        topic="delivery_delay",
        terms=("催发货", "一直不发货", "迟迟不发货", "延迟发货", "发货慢"),
        overview="催发货和延迟发货问题通常要结合下单时间、库存状态和页面承诺时效判断。",
        materials="建议准备订单号、下单时间、商品页面承诺时效截图以及当前订单状态。",
        timeline="发货时间通常受库存、活动高峰和仓库排队影响，实际出库时间要以订单状态更新为准。",
        fees="是否涉及补偿或运费调整通常要以平台规则和实际延迟原因判断。",
        eligibility="是否属于异常延迟，通常要对照商品页面承诺时效和平台规则确认。",
        process="建议先核对订单状态和承诺时效；若明显超时，可发起催发货或联系人工客服跟进。",
        contact="如果订单长时间未出库，建议携带订单号联系人工客服催单。",
    ),
    TopicRule(
        topic="warranty_period",
        terms=("保修期", "质保期", "保修多久", "质保多久"),
        overview="保修期或质保期通常要结合商品类目、品牌规则、购买时间和保修凭证确认。",
        materials="建议准备商品名称、型号、下单时间和保修凭证。",
        timeline="保修期本身属于规则信息，是否还在有效期内通常要根据购买时间和官方规则核对。",
        fees="若超出保修期，通常更可能涉及收费维修；具体费用仍要以售后检测结果为准。",
        eligibility="是否在保修范围内，通常取决于购买时间、故障类型和是否有人为损坏。",
        process="建议先核对页面、保修卡或说明书中的保修规则，再确认当前是否还在期限内。",
        contact="如果你拿不准保修期起止时间，建议带上订单号联系人工客服或官方售后核实。",
    ),
    TopicRule(
        topic="accessory_request",
        terms=("配件", "附件", "包装盒", "缺少配件", "少配件", "补寄配件"),
        overview="配件、附件、包装盒或补寄问题通常要先确认商品包装清单和实际缺少的内容。",
        materials="建议准备订单号、商品名称、缺少的具体配件名称以及开箱照片或视频。",
        timeline="补寄处理时效通常取决于售后审核、配件库存和物流安排。",
        fees="是否收费通常要看缺件原因、是否属于原包装缺失以及平台售后规则。",
        eligibility="是否支持补寄，通常要结合包装清单、问题责任和售后规则判断。",
        process="建议先核对商品包装清单，再提交缺件或补寄申请，并说明缺少的具体部件。",
        contact="如页面无法提交补寄，建议带上订单号联系人工客服处理。",
        scenarios=(
            ScenarioRule(
                name="accessory_rejected",
                terms=("补寄失败", "补件失败", "驳回", "拒绝补寄", "不给补寄", "审核不通过"),
                overview="如果补寄配件申请被驳回，通常要先看驳回原因，是包装清单不符、超出售后时效，还是证据不足。",
                materials="建议准备订单号、包装清单截图、缺少配件照片、开箱视频和售后单驳回截图。",
                timeline="补件驳回后的复核时效通常要看补充材料是否充分以及客服工单进度。",
                fees="若最终不支持免费补寄，是否能付费补购通常要看品牌配件供应和平台规则。",
                eligibility="是否还能重新申请，通常取决于是否仍在售后时效内，以及是否能补充更完整的缺件证据。",
                process="建议先核对驳回原因；如果是证据不足，就补充开箱视频和包装清单后重新提交；如果是规则限制，可改为咨询付费补购。",
                contact="如果你认为包装内确实缺件但申请被拒，建议带上包装清单和售后截图联系人工客服复核。",
            ),
        ),
    ),
)

_DETAIL_INTENT_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("materials", ("材料", "资料", "凭证", "截图", "证明", "准备什么", "需要什么", "提供什么")),
    ("timeline", ("多久", "几天", "多长时间", "什么时候", "何时", "多快", "时效")),
    ("fees", ("收费", "费用", "多少钱", "运费", "免费", "要钱", "花钱")),
    ("eligibility", ("可以吗", "能不能", "是否支持", "条件", "符合", "满足", "能否", "还能")),
    ("process", ("怎么办", "怎么处理", "流程", "怎么申请", "如何申请", "步骤", "怎么走")),
    ("contact", ("联系谁", "找谁", "人工客服", "客服电话", "找哪个部门")),
)


class CustomerServicePolicy:
    """Generate conservative but helpful replies for generic service scenarios."""

    def answer(self, question: str, *, context_topics: list[str] | None = None) -> PolicyResponse:
        normalized = question.strip()
        matched_rules = self._match_rules(normalized)
        detail_intents = self._detect_detail_intents(normalized)

        if not matched_rules and context_topics:
            matched_rules = [rule for rule in _TOPIC_RULES if rule.topic in context_topics]

        if not matched_rules:
            answer = (
                "这类问题更偏向订单、售后或平台服务流程，当前说明书资料无法直接确认。"
                "建议你补充订单号、商品名称、购买渠道、问题现象以及相关照片或聊天记录，"
                "这样更方便进一步判断应该走退款、换货、补发、维修还是投诉处理。"
            )
            return PolicyResponse(answer=answer, confidence=0.58, matched_topics=[])

        matched_scenarios = [self._match_scenario(rule, normalized) for rule in matched_rules[:2]]
        snippets = [
            self._compose_topic_answer(rule, detail_intents=detail_intents, scenario=scenario)
            for rule, scenario in zip(matched_rules[:2], matched_scenarios)
        ]
        answer = self._compose_answer(snippets, detail_intents=detail_intents)
        scenario_bonus = sum(1 for scenario in matched_scenarios if scenario is not None)
        confidence = min(
            0.7 + 0.03 * len({rule.topic for rule in matched_rules}) + 0.02 * len(detail_intents) + 0.01 * scenario_bonus,
            0.92,
        )
        return PolicyResponse(
            answer=answer,
            confidence=confidence,
            matched_topics=[rule.topic for rule in matched_rules],
        )

    def _match_rules(self, question: str) -> list[TopicRule]:
        matched: list[TopicRule] = []
        for rule in _TOPIC_RULES:
            if any(term in question for term in rule.terms):
                matched.append(rule)
        return matched

    def _detect_detail_intents(self, question: str) -> list[str]:
        intents: list[str] = []
        for intent, terms in _DETAIL_INTENT_RULES:
            if any(term in question for term in terms):
                intents.append(intent)
        return intents

    def _match_scenario(self, rule: TopicRule, question: str) -> ScenarioRule | None:
        best_match: ScenarioRule | None = None
        best_score = 0
        for scenario in rule.scenarios:
            score = sum(1 for term in scenario.terms if term in question)
            if score > best_score:
                best_match = scenario
                best_score = score
        return best_match

    def _pick_field(self, rule: TopicRule, scenario: ScenarioRule | None, field_name: str) -> str:
        if scenario is not None:
            scenario_text = getattr(scenario, field_name)
            if scenario_text:
                return str(scenario_text)
        return str(getattr(rule, field_name))

    def _compose_topic_answer(
        self,
        rule: TopicRule,
        *,
        detail_intents: list[str],
        scenario: ScenarioRule | None,
    ) -> str:
        overview = self._pick_field(rule, scenario, "overview")
        materials = self._pick_field(rule, scenario, "materials")
        timeline = self._pick_field(rule, scenario, "timeline")
        fees = self._pick_field(rule, scenario, "fees")
        eligibility = self._pick_field(rule, scenario, "eligibility")
        process = self._pick_field(rule, scenario, "process")
        contact = self._pick_field(rule, scenario, "contact")

        if not detail_intents:
            return " ".join((overview, materials, process))

        parts: list[str] = [overview]
        if "materials" in detail_intents:
            parts.append(materials)
        if "timeline" in detail_intents:
            parts.append(timeline)
        if "fees" in detail_intents:
            parts.append(fees)
        if "eligibility" in detail_intents:
            parts.append(eligibility)
        if "process" in detail_intents:
            parts.append(process)
        if "contact" in detail_intents:
            parts.append(contact)

        if len(parts) == 1:
            parts.extend((materials, process))
        return " ".join(parts)

    def _compose_answer(self, snippets: list[str], *, detail_intents: list[str]) -> str:
        unique_snippets: list[str] = []
        seen: set[str] = set()
        for snippet in snippets:
            if snippet in seen:
                continue
            seen.add(snippet)
            unique_snippets.append(snippet)

        closing = "如果你愿意，我建议下一步优先补充订单号、商品名称或型号、购买渠道，以及异常照片或截图。"
        if "contact" in detail_intents:
            closing = "如果你已经准备好了订单号和截图，建议直接联系人工客服，这样通常更快进入人工核查。"
        elif "timeline" in detail_intents:
            closing = "如果你愿意，我建议同时提供订单号和当前订单状态，这样更方便进一步判断实际处理时效。"
        elif "materials" in detail_intents:
            closing = "如果你现在方便，我建议先把订单号、截图或凭证信息整理好，再继续提交或联系人工客服。"

        return (
            "这类问题更适合按通用客服流程处理。"
            f"{' '.join(unique_snippets[:2])} "
            f"{closing}"
        ).strip()
