import torch
import torch.nn as nn


class TAN(nn.Module):
    def __init__(self, **model_args):
        super().__init__()
        self.timestamp_dim = model_args['timestamp_dim']
        self.timestamp_hidden = model_args['timestamp_hidden']
        self.is_xformer = model_args['is_xformer']

        if "time_of_day" in model_args['tan_timestamp']:
            self.time_of_day_size = model_args["time_of_day_size"]
            self.tod = nn.Embedding(num_embeddings=self.time_of_day_size, embedding_dim=self.timestamp_dim)
            torch.nn.init.normal_(self.tod.weight, mean=0.0, std=1.0)
        else:
            self.tod = None

        if "day_of_week" in model_args['tan_timestamp']:
            self.day_of_week_size = model_args["day_of_week_size"]
            self.dow = nn.Embedding(num_embeddings=self.day_of_week_size, embedding_dim=self.timestamp_dim)
            torch.nn.init.normal_(self.dow.weight, mean=0.0, std=1.0)
        else:
            self.dow = None

        if "day_of_month" in model_args['tan_timestamp']:
            self.day_of_month_size = model_args["day_of_month_size"]
            self.dom = nn.Embedding(num_embeddings=self.day_of_month_size, embedding_dim=self.timestamp_dim)
            torch.nn.init.normal_(self.dom.weight, mean=0.0, std=1.0)
        else:
            self.dom = None

        if "day_of_year" in model_args['tan_timestamp']:
            self.day_of_year_size = model_args["day_of_year_size"]
            self.doy = nn.Embedding(num_embeddings=self.day_of_year_size, embedding_dim=self.timestamp_dim)
            torch.nn.init.normal_(self.doy.weight, mean=0.0, std=1.0)
        else:
            self.doy = None

        #self.id = nn.Embedding(num_embeddings=model_args['dec_in'], embedding_dim=self.timestamp_dim)
        #torch.nn.init.normal_(self.id.weight, mean=0.0, std=1.0)

        self.transform_tod = nn.Linear(in_features=self.timestamp_dim, out_features=self.timestamp_dim, bias=False)
        self.transform_diw = nn.Linear(in_features=self.timestamp_dim, out_features=self.timestamp_dim, bias=False)

        self.times_w = nn.Sequential(
            nn.Linear(in_features=self.timestamp_dim, out_features=self.timestamp_hidden),
            nn.Tanh(),
            nn.Linear(in_features=self.timestamp_hidden, out_features=self.timestamp_dim, bias=False),
        )
        self.times_b = nn.Sequential(
            nn.Linear(in_features=self.timestamp_dim, out_features=self.timestamp_hidden),
            nn.Tanh(),
            nn.Linear(in_features=self.timestamp_hidden, out_features=self.timestamp_dim, bias=False),
        )



        # Similarly initialize dom/doy embeddings if needed...

    def forward(self, x_mark_enc, x_mark_dec):
        """Process timestamp embeddings and return transformation parameters"""
        conet_s_lst, conet_e_lst = [], []

        # Process each timestamp type
        if self.tod:
            tod_s = x_mark_enc[:, 0, 0].detach()
            tod_e = x_mark_dec[:, 0, 0].detach()
            tod_s = ((tod_s + 0.5) * self.time_of_day_size).to(torch.int64)
            tod_e = ((tod_e + 0.5) * self.time_of_day_size).to(torch.int64)
            '''if self.is_xformer:
                tod_s = ((tod_s + 0.5) * self.time_of_day_size).to(torch.int64)
                tod_e = ((tod_e + 0.5) * self.time_of_day_size).to(torch.int64)
            else:
                tod_s = (tod_s * self.time_of_day_size).to(torch.int64)
                tod_e = (tod_e * self.time_of_day_size).to(torch.int64)'''
            conet_tod_s = self.tod(tod_s)
            conet_tod_e = self.tod(tod_e)
            conet_s_lst.append(conet_tod_s)
            conet_e_lst.append(conet_tod_e)

        if self.dow:
            dow_s = x_mark_enc[:, 0, 1].detach()
            dow_e = x_mark_dec[:, 0, 1].detach()
            dow_s = ((dow_s + 0.5) * self.day_of_week_size).to(torch.int64)
            dow_e = ((dow_e + 0.5) * self.day_of_week_size).to(torch.int64)
            '''if self.is_xformer:
                dow_s = ((dow_s + 0.5) * self.day_of_week_size).to(torch.int64)
                dow_e = ((dow_e + 0.5) * self.day_of_week_size).to(torch.int64)
            else:
                dow_s = (dow_s * self.day_of_week_size).to(torch.int64)
                dow_e = (dow_e * self.day_of_week_size).to(torch.int64)'''
            conet_dow_s = self.dow(dow_s)
            conet_dow_e = self.dow(dow_e)
            conet_s_lst.append(conet_dow_s)
            conet_e_lst.append(conet_dow_e)

        if self.dom:
            dom_s = x_mark_enc[:, 0, 2].detach()
            dom_e = x_mark_dec[:, 0, 2].detach()
            dom_s = ((dom_s + 0.5) * self.day_of_month_size).to(torch.int64)
            dom_e = ((dom_e + 0.5) * self.day_of_month_size).to(torch.int64)
            '''if self.is_xformer:
                dom_s = ((dom_s + 0.5) * self.day_of_month_size).to(torch.int64)
                dom_e = ((dom_e + 0.5) * self.day_of_month_size).to(torch.int64)
            else:
                dom_s = (dom_s * self.day_of_month_size).to(torch.int64)
                dom_e = (dom_e * self.day_of_month_size).to(torch.int64)'''
            conet_dom_s = self.dom(dom_s)
            conet_dom_e = self.dom(dom_e)
            conet_s_lst.append(conet_dom_s)
            conet_e_lst.append(conet_dom_e)

        if self.doy:
            doy_s = x_mark_enc[:, 0, 3].detach()
            doy_e = x_mark_dec[:, 0, 3].detach()
            doy_s = ((doy_s + 0.5) * self.day_of_year_size).to(torch.int64)
            doy_e = ((doy_e + 0.5) * self.day_of_year_size).to(torch.int64)
            '''if self.is_xformer:
                doy_s = ((doy_s + 0.5) * self.day_of_year_size).to(torch.int64)
                doy_e = ((doy_e + 0.5) * self.day_of_year_size).to(torch.int64)
            else:
                doy_s = (doy_s * self.day_of_year_size).to(torch.int64)
                doy_e = (doy_e * self.day_of_year_size).to(torch.int64)'''
            conet_doy_s = self.doy(doy_s)
            conet_doy_e = self.doy(doy_e)
            conet_s_lst.append(conet_doy_s)
            conet_e_lst.append(conet_doy_e)

        # Combine embeddings
        conet_s = torch.sum(torch.stack(conet_s_lst), dim=0) / torch.sqrt(torch.tensor(len(conet_s_lst)))
        conet_e = torch.sum(torch.stack(conet_e_lst), dim=0) / torch.sqrt(torch.tensor(len(conet_e_lst)))

        conet_s_w = self.times_w(conet_s)
        conet_s_b = self.times_b(conet_s)
        times_w = torch.sum(conet_s_w * conet_e / torch.sqrt(torch.tensor(self.timestamp_dim)), dim=1,
                            keepdim=True).unsqueeze(-1)
        times_b = torch.sum(conet_s_b * conet_e / torch.sqrt(torch.tensor(self.timestamp_dim)), dim=1,
                            keepdim=True).unsqueeze(-1)
        return times_w, times_b

    def get_combined_embeddings(self):
        time_embedding_list = []
        if self.tod:
            tod_size = torch.tensor([i for i in range(self.time_of_day_size)]).to(self.tod.weight.device)
            time_embedding_list.append(self.tod(tod_size))
        if self.dow:
            dow_size = torch.tensor([i for i in range(self.day_of_week_size)]).to(self.dow.weight.device)
            time_embedding_list.append(self.dow(dow_size))
        if self.dom:
            dom_size = torch.tensor([i for i in range(self.day_of_month_size)]).to(self.dom.weight.device)
            time_embedding_list.append(self.dom(dom_size))
        if self.doy:
            doy_size = torch.tensor([i for i in range(self.day_of_year_size)]).to(self.doy.weight.device)
            time_embedding_list.append(self.doy(doy_size))
        time_embedding = torch.cat(time_embedding_list, dim=0)
        return time_embedding